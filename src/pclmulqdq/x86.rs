// Copyright 2020 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! x86/x86_64 implementation of the PCLMULQDQ-based CRC calculation.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::BitXor;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Simd(__m128i);

impl super::SimdExt for Simd {
    fn is_supported() -> bool {
        is_x86_feature_detected!("pclmulqdq") // _mm_clmulepi64_si128
            && is_x86_feature_detected!("sse2") // (all other _mm_*)
            && is_x86_feature_detected!("sse4.1") // _mm_extract_epi64
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn new(high: u64, low: u64) -> Self {
        Self(_mm_set_epi64x(high as i64, low as i64))
    }

    #[inline]
    #[target_feature(enable = "sse2", enable = "pclmulqdq")]
    unsafe fn fold_16(self, coeff: Self) -> Self {
        let h = Self(_mm_clmulepi64_si128(self.0, coeff.0, 0x11));
        let l = Self(_mm_clmulepi64_si128(self.0, coeff.0, 0x00));
        h ^ l
    }

    #[inline]
    #[target_feature(enable = "sse2", enable = "pclmulqdq")]
    unsafe fn fold_8(self, coeff: u64) -> Self {
        let coeff = Self::new(0, coeff);
        let h = Self(_mm_clmulepi64_si128(self.0, coeff.0, 0x00));
        let l = Self(_mm_srli_si128(self.0, 8));
        h ^ l
    }

    #[inline]
    #[target_feature(enable = "sse2", enable = "sse4.1", enable = "pclmulqdq")]
    unsafe fn barrett(self, poly: u64, mu: u64) -> u64 {
        let polymu = Self::new(poly, mu);
        let t1 = _mm_clmulepi64_si128(self.0, polymu.0, 0x00);
        let h = Self(_mm_slli_si128(t1, 8));
        let l = Self(_mm_clmulepi64_si128(t1, polymu.0, 0x10));
        let reduced = h ^ l ^ self;
        _mm_extract_epi64(reduced.0, 1) as u64
    }
}

impl BitXor for Simd {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        Self(unsafe { _mm_xor_si128(self.0, other.0) })
    }
}
