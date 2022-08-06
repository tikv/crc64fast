// Copyright 2019 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! PCLMULQDQ-based CRC-64-ECMA computer.
//!
//! The implementation is based on Intel's "Fast CRC Computation for Generic
//! Polynomials Using PCLMULQDQ Instruction" [white paper].
//!
//! [white paper]: https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf

#[cfg(not(feature = "fake-simd"))]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), path = "x86.rs")]
#[cfg_attr(all(target_arch = "aarch64"), path = "aarch64.rs")]
mod arch;

#[cfg(feature = "fake-simd")]
mod arch;

use self::arch::Simd;
use super::table;
use std::{
    fmt::Debug,
    ops::{BitXor, BitXorAssign},
};

/// This trait must be implemented on `self::arch::Simd` to provide the
/// platform-specific SIMD implementations.
trait SimdExt: Copy + Debug + BitXor {
    /// Returns whether SIMD-accelerated carryless multiplication is supported.
    fn is_supported() -> bool;

    /// Creates a new 128-bit integer from the 64-bit parts.
    unsafe fn new(high: u64, low: u64) -> Self;

    /// Performs a CRC folding step across 16 bytes.
    ///
    /// Should return `(coeff.low_64 ⊗ self.low_64) ⊕ (coeff.high_64 ⊗ self.high_64)`,
    /// where ⊕ is XOR and ⊗ is carryless multiplication.
    unsafe fn fold_16(self, coeff: Self) -> Self;

    /// Performs a CRC folding step across 8 bytes.
    ///
    /// Should return `self.high_64 ⊕ (coeff ⊗ self.low_64)`,
    /// where ⊕ is XOR and ⊗ is carryless multiplication.
    unsafe fn fold_8(self, coeff: u64) -> Self;

    /// Performs Barrett reduction to finalize the CRC.
    ///
    /// Should return `(self ⊕ ((self.low_64 ⊗ mu).low_64 ⊗ (poly ⊕ 2^64))).high_64`,
    /// where ⊕ is XOR and ⊗ is carryless multiplication.
    unsafe fn barrett(self, poly: u64, mu: u64) -> u64;
}

impl PartialEq for Simd {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            use std::mem::transmute;
            let a: u128 = transmute(*self);
            let b: u128 = transmute(*other);
            a == b
        }
    }
}

impl Eq for Simd {}

impl BitXorAssign for Simd {
    fn bitxor_assign(&mut self, other: Self) {
        *self = *self ^ other;
    }
}

pub fn get_update() -> super::UpdateFn {
    if Simd::is_supported() {
        update
    } else {
        table::update
    }
}

fn update(mut state: u64, bytes: &[u8]) -> u64 {
    let (left, middle, right) = unsafe { bytes.align_to::<[Simd; 8]>() };
    if let Some((first, rest)) = middle.split_first() {
        state = table::update(state, left);
        state = unsafe { update_simd(state, first, rest) };
        table::update(state, right)
    } else {
        table::update(state, bytes)
    }
}

#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")
)]
#[cfg_attr(all(target_arch = "aarch64"), target_feature(enable = "neon,aes"))]
unsafe fn update_simd(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]]) -> u64 {
    // receive the initial 128 bytes of data
    let mut x = *first;

    // xor the initial CRC value
    x[0] ^= Simd::new(0, state);

    // perform 128-byte folding.
    let coeff = Simd::new(table::K_1023, table::K_1087);
    for chunk in rest {
        for (xi, yi) in x.iter_mut().zip(chunk.iter()) {
            *xi = *yi ^ xi.fold_16(coeff);
        }
    }

    let coeffs = [
        Simd::new(table::K_895, table::K_959), // fold by distance of 112 bytes
        Simd::new(table::K_767, table::K_831), // fold by distance of 96 bytes
        Simd::new(table::K_639, table::K_703), // fold by distance of 80 bytes
        Simd::new(table::K_511, table::K_575), // fold by distance of 64 bytes
        Simd::new(table::K_383, table::K_447), // fold by distance of 48 bytes
        Simd::new(table::K_255, table::K_319), // fold by distance of 32 bytes
        Simd::new(table::K_127, table::K_191), // fold by distance of 16 bytes
    ];
    x.iter()
        .zip(&coeffs)
        .fold(x[7], |acc, (m, c)| acc ^ m.fold_16(*c))
        .fold_8(table::K_127) // finally fold 16 bytes into 8 bytes.
        .barrett(table::POLY, table::MU) // barrett reduction.
}

#[test]
fn test_size_and_alignment() {
    assert_eq!(std::mem::size_of::<Simd>(), 16);
    assert_eq!(std::mem::align_of::<Simd>(), 16);
}

#[test]
fn test_new() {
    unsafe {
        let x = Simd::new(0xd7c8_11cf_e5c5_c792, 0x86e6_5c36_e68b_4804);
        let y = Simd::new(0xd7c8_11cf_e5c5_c792, 0x86e6_5c36_e68b_4804);
        let z = Simd::new(0xfa3e_0099_cd5e_d60d, 0xad71_9ee6_57d1_498e);
        assert_eq!(x, y);
        assert_ne!(x, z);
    }
}

#[test]
fn test_xor() {
    unsafe {
        let x = Simd::new(0xe450_87f9_b031_0d47, 0x3d72_e92a_96c7_4c63);
        let y = Simd::new(0x7ed8_ae0a_dfbd_89c0, 0x1c9b_dfaa_953e_0ef4);
        let mut z = x ^ y;
        assert_eq!(z, Simd::new(0x9a88_29f3_6f8c_8487, 0x21e9_3680_03f9_4297));
        z ^= Simd::new(0x57a2_0f44_c005_b2ea, 0x7056_bde9_9303_aa51);
        assert_eq!(z, Simd::new(0xcd2a_26b7_af89_366d, 0x51bf_8b69_90fa_e8c6));
    }
}

#[test]
fn test_fold_16() {
    unsafe {
        let x = Simd::new(0xb5f1_2590_5645_0b6c, 0x333a_2c49_c361_9e21);
        let f = x.fold_16(Simd::new(0xbecc_9dd9_038f_c366, 0x5ba9_365b_e2e9_5bf5));
        assert_eq!(f, Simd::new(0x4f55_42df_ef35_1810, 0x0c03_5bd6_70fc_5abd));
    }
}

#[test]
fn test_fold_8() {
    unsafe {
        let x = Simd::new(0x60c0_b48f_4a92_2003, 0x203c_f7bc_ad34_103b);
        let f = x.fold_8(0x3e90_3688_ea71_f472);
        assert_eq!(f, Simd::new(0x07d7_2761_4d16_56db, 0x2bc0_ed8a_a341_7665));
    }
}

#[test]
fn test_barrett() {
    unsafe {
        let x = Simd::new(0x2606_e582_3406_9bae, 0x76cc_1105_0fef_6d68);
        let b = x.barrett(0x435d_0f79_19a6_1445, 0x5817_6272_f8fa_b8d5);
        assert_eq!(b, 0x5e4d_0253_942a_d95d);
    }
}
