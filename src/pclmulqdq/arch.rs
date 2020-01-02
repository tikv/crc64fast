// Copyright 2020 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! A platform-agnostic implementation of the PCLMULQDQ-based CRC calculation.
//!
//! This is used to compare against the platform-specific implementations.
//! Enable the `fake-simd` feature to use this. Note that this implementation is
//! 100× slower than a real SIMD implementation, and should never be used in
//! production code.

use std::ops::BitXor;

#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct Simd(u128);

impl super::SimdExt for Simd {
    fn is_supported() -> bool {
        cfg!(feature = "fake-simd")
    }

    unsafe fn new(high: u64, low: u64) -> Self {
        Simd(u128::from(low) | u128::from(high) << 64)
    }

    unsafe fn fold_16(self, coeff: Self) -> Self {
        let h = poly_mul(coeff.0 as u64, self.0 as u64);
        let l = poly_mul((coeff.0 >> 64) as u64, (self.0 >> 64) as u64);
        Self(h ^ l)
    }

    unsafe fn fold_8(self, coeff: u64) -> Self {
        let h = poly_mul(coeff, self.0 as u64);
        let l = self.0 >> 64;
        Self(h ^ l)
    }

    unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
        let t1 = poly_mul(self.0 as u64, mu);
        let h = t1 << 64;
        let l = poly_mul(t1 as u64, poly);
        let reduced = self.0 ^ h ^ l;
        (reduced >> 64) as u64
    }
}

impl BitXor for Simd {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}

fn poly_mul(a: u64, b: u64) -> u128 {
    let mut res = 0;
    for i in 0..64 {
        if a & (1 << i) != 0 {
            res ^= u128::from(b) << i;
        }
    }
    res
}

#[test]
fn test_poly_mul() {
    assert_eq!(
        poly_mul(0x5a2d_8244_0f1e_3e50, 0xcae9_00d5_fed9_262f),
        0x39ca_c5ca_fc66_6bf3_25bc_9dd4_c0f3_6330,
    )
}
