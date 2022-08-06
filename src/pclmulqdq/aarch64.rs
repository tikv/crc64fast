// Copyright 2020 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! AArch64 implementation of the PCLMULQDQ-based CRC calculation.

use std::arch::aarch64::*;
use std::arch::asm;
use std::arch::is_aarch64_feature_detected;
use std::ops::BitXor;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Simd(uint8x16_t);

impl Simd {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn from_mul(a: u64, b: u64) -> Self {
        let mul = vmull_p64(a, b);
        Self(vreinterpretq_u8_p128(mul))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn into_poly64s(self) -> [u64; 2] {
        let x = vreinterpretq_p64_u8(self.0);
        [vgetq_lane_p64(x, 0), vgetq_lane_p64(x, 1)]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn high_64(self) -> u64 {
        let x = vreinterpretq_p64_u8(self.0);
        vgetq_lane_p64(x, 1)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn low_64(self) -> u64 {
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
    #[target_feature(enable = "neon,aes")]
    unsafe fn fold_16(self, coeff: Self) -> Self {
        let h: Self;
        let l: Self;

        // FIXME: When used as a single function, this branch is equivalent to
        // the ASM below. However, when fold_16 is called inside a loop, for
        // some reason LLVM replaces the PMULL2 call with a plain PMULL, which
        // leads unnecessary FMOV calls and slows down the throughput by about
        // 20-25%. This bug does not exist with GCC. Delete the
        // ASM code once this misoptimization is fixed.
        #[cfg(slow)]
        {
            let [x0, x1] = self.into_poly64s();
            let [c0, c1] = coeff.into_poly64s();
            h = Self::from_mul(c0, x0);
            l = Self::from_mul(c1, x1);
        }
        #[cfg(not(slow))]
        #[allow(asm_sub_register)]
        {
            let temp_l: uint8x16_t;
            let temp_h: uint8x16_t;
            asm!(
                "pmull {low}.1q, {in1}.1d, {in2}.1d",
                "pmull2 {high}.1q, {in1}.2d, {in2}.2d",
                low = out(vreg) temp_l,
                high = out(vreg) temp_h,
                in1 = in(vreg) self.0,
                in2 = in(vreg) coeff.0,
            );
            l = Self(temp_l);
            h = Self(temp_h);
        }

        h ^ l
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn fold_8(self, coeff: u64) -> Self {
        let [x0, x1] = self.into_poly64s();
        let h = Self::from_mul(coeff, x0);
        let l = Self::new(0, x1);
        h ^ l
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
        let t1 = Self::from_mul(self.low_64(), mu).low_64();
        let l = Self::from_mul(t1, poly);
        let reduced: u64 = (self ^ l).high_64();
        let t1: u64 = t1;
        reduced ^ t1
    }
}

impl BitXor for Simd {
    type Output = Simd;

    fn bitxor(self, other: Self) -> Self {
        unsafe { Self(veorq_u8(self.0, other.0)) }
    }
}
