// Copyright 2020 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! AArch64 implementation of the PCLMULQDQ-based CRC calculation.

use std::arch::{aarch64::*, is_aarch64_feature_detected};
use std::mem::transmute;
use std::ops::BitXor;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Simd(uint8x16_t);

#[allow(non_camel_case_types)]
type poly64_t = u64;

impl Simd {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn from_mul(a: poly64_t, b: poly64_t) -> Self {
        let mul = vmull_p64(a, b);
        Self(vreinterpretq_u8_p128(mul))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn into_poly64s(self) -> [poly64_t; 2] {
        let x = vreinterpretq_p64_u8(self.0);
        [vgetq_lane_p64(x, 0), vgetq_lane_p64(x, 1)]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn high_64(self) -> poly64_t {
        let x = vreinterpretq_p64_u8(self.0);
        vgetq_lane_p64(x, 1)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn low_64(self) -> poly64_t {
        let x = vreinterpretq_p64_u8(self.0);
        vgetq_lane_p64(x, 0)
    }
}

impl super::SimdExt for Simd {
    fn is_supported() -> bool {
        is_aarch64_feature_detected!("pmull") && is_aarch64_feature_detected!("neon")
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn new(high: u64, low: u64) -> Self {
        Self(vcombine_u8(vcreate_u8(low), vcreate_u8(high)))
    }

    #[inline]
    #[target_feature(enable = "aes", enable = "neon")]
    unsafe fn fold_16(self, coeff: Self) -> Self {
        let [x0, x1] = self.into_poly64s();
        let [c0, c1] = coeff.into_poly64s();
        let h = Self::from_mul(c0, x0);
        let l = Self::from_mul(c1, x1);
        h ^ l
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn fold_8(self, coeff: u64) -> Self {
        let [x0, x1] = self.into_poly64s();
        let h = Self::from_mul(transmute(coeff), x0);
        let l = Self::new(0, transmute(x1));
        h ^ l
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
        let t1 = Self::from_mul(self.low_64(), transmute(mu)).low_64();
        let l = Self::from_mul(t1, transmute(poly));
        let reduced: u64 = transmute((self ^ l).high_64());
        let t1: u64 = transmute(t1);
        reduced ^ t1
    }
}

impl BitXor for Simd {
    type Output = Simd;

    fn bitxor(self, other: Self) -> Self {
        unsafe { Self(veorq_u8(self.0, other.0)) }
    }
}
