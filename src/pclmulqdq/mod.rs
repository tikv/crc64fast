// Copyright 2019 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! PCLMULQDQ-based CRC-64-ECMA computer.
//!
//! The implementation is based on Intel's "Fast CRC Computation for Generic
//! Polynomials Using PCLMULQDQ Instruction" [white paper].
//!
//! [white paper]: https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf

#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), path = "x86.rs")]
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
    /// Should return `(high ⊗ self.low_64) ⊕ (low ⊗ self.high_64)`,
    /// where ⊕ is XOR and ⊗ is carryless multiplication.
    unsafe fn fold_16(self, high: u64, low: u64) -> Self;

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
unsafe fn update_simd(state: u64, first: &[Simd; 8], rest: &[[Simd; 8]]) -> u64 {
    // receive the initial 128 bytes of data
    let mut x = *first;

    // xor the initial CRC value
    x[0] ^= Simd::new(0, state);

    // perform 128-byte folding.
    for chunk in rest {
        for (xi, yi) in x.iter_mut().zip(chunk.iter()) {
            *xi = *yi ^ xi.fold_16(table::K_1087, table::K_1023);
        }
    }

    let x = x[0].fold_16(table::K_959, table::K_895) // fold by distance of 112 bytes
        ^ x[1].fold_16(table::K_831, table::K_767) // fold by distance of 96 bytes
        ^ x[2].fold_16(table::K_703, table::K_639) // fold by distance of 80 bytes
        ^ x[3].fold_16(table::K_575, table::K_511) // fold by distance of 64 bytes
        ^ x[4].fold_16(table::K_447, table::K_383) // fold by distance of 48 bytes
        ^ x[5].fold_16(table::K_319, table::K_255) // fold by distance of 32 bytes
        ^ x[6].fold_16(table::K_191, table::K_127) // fold by distance of 16 bytes
        ^ x[7];

    // finally fold 16 bytes into 8 bytes.
    let r = x.fold_8(table::K_127);

    // barrett reduction.
    r.barrett(table::POLY, table::MU)
}
