//! 64-bit image atomics (`dmask` 0x3, cmpswap 0xF): lowered to the i64
//! overloads of DXIL's atomic operations on a u64 typed UAV, with the
//! feature bits the host must support. Runs without a GPU; dxil.dll, where
//! there is one, validates the modules.

use super::{lower, shader::features};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// `image_atomic_<op>` of 2D image s[4:11] at (v4, v5) with data at v0.
fn image_atomic(op: u32, dmask: u32, glc: bool) -> [u32; 2] {
    let w0 = 0xF000_0000 | op << 18 | u32::from(glc) << 13 | dmask << 8 | 1 << 3;
    let w1 = 1 << 16 | 4; // srsrc s4, vdata v0, vaddr v4
    [w0, w1]
}

/// A wave32 compute program with the table at s0 whose first entry is the
/// UAV the atomic names; data v[0:3] = (5, 1, 7, 2), coordinates (tid, 0).
fn lower_atomic(atomic: [u32; 2]) -> Result<Vec<u8>, ShaderError> {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF40C_0101,
        0xFA00_0000, // s_load_dwordx8 s[4:11], s[2:3], null
        0x7E08_0300, // v_mov_b32 v4, v0
        0x7E0A_0280, // v_mov_b32 v5, 0
        0x7E00_0285, // v_mov_b32 v0, 5
        0x7E02_0281, // v_mov_b32 v1, 1
        0x7E04_0287, // v_mov_b32 v2, 7
        0x7E06_0282, // v_mov_b32 v3, 2
        0xBF8C_0000, // s_waitcnt 0
    ];
    words.extend(atomic);
    words.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut lowered.dxil.clone())
            .expect("dxil.dll rejects the module");
    }
    Ok(lowered.dxil)
}

fn sfi0(dxil: &[u8]) -> u64 {
    let c = Container::parse(dxil).unwrap();
    let d = c.part(FourCC::SFI0).map_or(&[0u8; 8][..], |p| p.data);
    u64::from_le_bytes(d[..8].try_into().unwrap())
}

const WIDE: u64 = features::ATOMIC_INT64_TYPED | features::ATOMIC_INT64_HEAP;

#[test]
fn a_32_bit_image_atomic_needs_no_64_bit_features() {
    let dxil = lower_atomic(image_atomic(0x11, 0x1, true)).expect("lower");
    assert_eq!(sfi0(&dxil) & WIDE, 0);
}

#[test]
fn a_64_bit_image_atomic_lowers_and_declares_its_features() {
    for (op, name) in [
        (0x11, "add"),
        (0x12, "sub"),
        (0x14, "smin"),
        (0x15, "umin"),
        (0x16, "smax"),
        (0x17, "umax"),
        (0x18, "and"),
        (0x19, "or"),
        (0x1A, "xor"),
        (0x0F, "swap"),
    ] {
        for glc in [false, true] {
            let dxil = lower_atomic(image_atomic(op, 0x3, glc))
                .unwrap_or_else(|e| panic!("image_atomic_{name} glc={glc}: {e}"));
            let f = sfi0(&dxil);
            assert_eq!(f & WIDE, WIDE, "image_atomic_{name}: SFI0 {f:#x}");
            assert_ne!(f & 0x8000, 0, "image_atomic_{name}: no INT64_OPS");
        }
    }
}

#[test]
fn a_64_bit_image_cmpswap_lowers() {
    let dxil = lower_atomic(image_atomic(0x10, 0xF, true)).expect("lower");
    assert_eq!(sfi0(&dxil) & WIDE, WIDE);
    // The 32-bit cmpswap (two dwords) stays 32-bit.
    let dxil = lower_atomic(image_atomic(0x10, 0x3, true)).expect("lower");
    assert_eq!(sfi0(&dxil) & WIDE, 0);
}

#[test]
fn other_atomic_channel_masks_stay_unsupported() {
    for dmask in [0x7, 0x2] {
        let e = lower_atomic(image_atomic(0x11, dmask, false)).unwrap_err();
        assert!(
            matches!(
                e,
                ShaderError::Unsupported {
                    what: "an image atomic of these channels",
                    ..
                }
            ),
            "dmask {dmask:#x}: {e}"
        );
    }
}
