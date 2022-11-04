use super::{super::fold_tail, Simd, __cpuid_count, __m256i, _mm256_set_epi64x, _mm256_xor_si256};
use core::ops::BitXor;
use lazy_static::lazy_static;

// PCLMULQDQ can be used without avx512vl. However, this is only addressed by rust recently --- so we
// need to manually specify the intrinsic, otherwise rustc will inline it poorly.
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.pclmulqdq.256"]
    fn pclmulqdq_256(a: __m256i, round_key: __m256i, imm8: u8) -> __m256i;
}

#[derive(Clone, Copy, Debug)]
pub struct Simd256(__m256i);

lazy_static! {
    static ref VPCLMULQDQ_SUPPORTED : bool = {
        let avx2 = is_x86_feature_detected!("avx2");
        // Rust is very confused about VPCLMULQDQ
        // Let us detect it use CPUID directly
        let leaf_7 = unsafe { __cpuid_count(7, 0) };
        let vpclmulqdq = (leaf_7.ecx & (1u32 << 10)) != 0;
        avx2 && vpclmulqdq
    };
}

impl Simd256 {
    #[inline]
    pub fn is_supported() -> bool {
        *VPCLMULQDQ_SUPPORTED
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(x3: u64, x2: u64, x1: u64, x0: u64) -> Self {
        Self(_mm256_set_epi64x(x3 as _, x2 as _, x1 as _, x0 as _))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn to_simd_x8(self4: [Self; 4]) -> [Simd; 8] {
        core::mem::transmute(self4)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "avx512vpclmulqdq")]
    pub unsafe fn fold_32(self, coeff: Self) -> Self {
        let h = pclmulqdq_256(self.0, coeff.0, 0x11);
        let l = pclmulqdq_256(self.0, coeff.0, 0x00);
        Self(h) ^ Self(l)
    }
}

impl BitXor for Simd256 {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, other: Self) -> Self {
        Self(unsafe { _mm256_xor_si256(self.0, other.0) })
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "avx512vpclmulqdq")]
pub(crate) unsafe fn update_vpclmulqdq(
    state: u64,
    first: &[[Simd256; 4]; 2],
    rest: &[[[Simd256; 4]; 2]],
) -> u64 {
    // receive the initial 128 bytes of data
    let (mut x, y) = (first[0], first[1]);

    // xor the initial CRC value
    x[0] = x[0] ^ Simd256::new(0, 0, 0, state);

    let coeff = Simd256::new(
        crate::table::K_1023,
        crate::table::K_1087,
        crate::table::K_1023,
        crate::table::K_1087,
    );

    x[0] = x[0].fold_32(coeff) ^ y[0];
    x[1] = x[1].fold_32(coeff) ^ y[1];
    x[2] = x[2].fold_32(coeff) ^ y[2];
    x[3] = x[3].fold_32(coeff) ^ y[3];

    // perform 256-byte folding.
    for chunk in rest {
        let chunk = *chunk;
        x[0] = x[0].fold_32(coeff) ^ chunk[0][0];
        x[0] = x[0].fold_32(coeff) ^ chunk[1][0];
        x[1] = x[1].fold_32(coeff) ^ chunk[0][1];
        x[1] = x[1].fold_32(coeff) ^ chunk[1][1];
        x[2] = x[2].fold_32(coeff) ^ chunk[0][2];
        x[2] = x[2].fold_32(coeff) ^ chunk[1][2];
        x[3] = x[3].fold_32(coeff) ^ chunk[0][3];
        x[3] = x[3].fold_32(coeff) ^ chunk[1][3];
    }

    let x = Simd256::to_simd_x8(x);
    fold_tail(x)
}

impl PartialEq for Simd256 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            use core::mem::transmute;
            let a: [u128; 2] = transmute(*self);
            let b: [u128; 2] = transmute(*other);
            a == b
        }
    }
}

impl Eq for Simd256 {}

#[cfg(target_feature = "avx2")]
#[test]
fn test_size_and_alignment() {
    assert_eq!(std::mem::size_of::<Simd256>(), 32);
    assert_eq!(std::mem::align_of::<Simd256>(), 32);
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_new() {
    unsafe {
        let x = Simd256::new(
            0xd7c8_11cf_e5c5_c792,
            0x86e6_5c36_e68b_4804,
            0xd7c8_11cf_e5c5_c792,
            0x86e6_5c36_e68b_4804,
        );
        let y = Simd256::new(
            0xd7c8_11cf_e5c5_c792,
            0x86e6_5c36_e68b_4804,
            0xd7c8_11cf_e5c5_c792,
            0x86e6_5c36_e68b_4804,
        );
        let z = Simd256::new(
            0xfa3e_0099_cd5e_d60d,
            0xad71_9ee6_57d1_498e,
            0xfa3e_0099_cd5e_d60d,
            0xad71_9ee6_57d1_498e,
        );
        assert_eq!(x, y);
        assert_ne!(x, z);
    }
}

#[cfg(target_feature = "avx2")]
#[test]
fn test_xor() {
    unsafe {
        let x = Simd256::new(
            0xe450_87f9_b031_0d47,
            0x3d72_e92a_96c7_4c63,
            0xe450_87f9_b031_0d47,
            0x3d72_e92a_96c7_4c63,
        );
        let y = Simd256::new(
            0x7ed8_ae0a_dfbd_89c0,
            0x1c9b_dfaa_953e_0ef4,
            0x7ed8_ae0a_dfbd_89c0,
            0x1c9b_dfaa_953e_0ef4,
        );
        let mut z = x ^ y;
        assert_eq!(
            z,
            Simd256::new(
                0x9a88_29f3_6f8c_8487,
                0x21e9_3680_03f9_4297,
                0x9a88_29f3_6f8c_8487,
                0x21e9_3680_03f9_4297
            )
        );
        z = z ^ Simd256::new(
            0x57a2_0f44_c005_b2ea,
            0x7056_bde9_9303_aa51,
            0x57a2_0f44_c005_b2ea,
            0x7056_bde9_9303_aa51,
        );
        assert_eq!(
            z,
            Simd256::new(
                0xcd2a_26b7_af89_366d,
                0x51bf_8b69_90fa_e8c6,
                0xcd2a_26b7_af89_366d,
                0x51bf_8b69_90fa_e8c6
            )
        );
    }
}

#[cfg(all(target_feature = "avx2", target_feature = "avx512vpclmulqdq"))]
#[test]
fn test_fold_32() {
    unsafe {
        let x = Simd256::new(
            0xb5f1_2590_5645_0b6c,
            0x333a_2c49_c361_9e21,
            0xb5f1_2590_5645_0b6c,
            0x333a_2c49_c361_9e21,
        );
        let f = x.fold_32(Simd256::new(
            0xbecc_9dd9_038f_c366,
            0x5ba9_365b_e2e9_5bf5,
            0xbecc_9dd9_038f_c366,
            0x5ba9_365b_e2e9_5bf5,
        ));
        assert_eq!(
            f,
            Simd256::new(
                0x4f55_42df_ef35_1810,
                0x0c03_5bd6_70fc_5abd,
                0x4f55_42df_ef35_1810,
                0x0c03_5bd6_70fc_5abd
            )
        );
    }
}
