// Copyright 2019 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! PCLMULQDQ-based CRC-64-ECMA computer.
//!
//! The implementation is based on Intel's "Fast CRC Computation for Generic
//! Polynomials Using PCLMULQDQ Instruction" [white paper].
//!
//! [white paper]: https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::table;

pub fn get_update() -> super::UpdateFn {
    if is_x86_feature_detected!("pclmulqdq") // _mm_clmulepi64_si128
        && is_x86_feature_detected!("sse2") // (all other _mm_*)
        && is_x86_feature_detected!("sse4.1")
    // _mm_extract_epi64
    {
        update
    } else {
        table::update
    }
}

fn update(mut state: u64, bytes: &[u8]) -> u64 {
    let (left, middle, right) = unsafe { bytes.align_to::<[__m128i; 8]>() };
    if let Some((first, rest)) = middle.split_first() {
        state = table::update(state, left);
        state = unsafe { update_simd(state, first, rest) };
        table::update(state, right)
    } else {
        table::update(state, bytes)
    }
}

#[target_feature(enable = "pclmulqdq", enable = "sse2")]
unsafe fn fold(coeff: __m128i, x: __m128i, y: __m128i) -> __m128i {
    let h = _mm_clmulepi64_si128(x, coeff, 0x10);
    let l = _mm_clmulepi64_si128(x, coeff, 0x01);
    _mm_xor_si128(_mm_xor_si128(h, l), y)
}

#[target_feature(enable = "sse2")]
unsafe fn build_const(high: u64, low: u64) -> __m128i {
    _mm_set_epi64x(high as i64, low as i64)
}

#[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
unsafe fn update_simd(state: u64, first: &[__m128i; 8], rest: &[[__m128i; 8]]) -> u64 {
    let state = build_const(0, state);

    // receive the initial 128 bytes of data
    let mut x0 = _mm_load_si128(first.as_ptr());
    let mut x1 = _mm_load_si128(first.as_ptr().add(1));
    let mut x2 = _mm_load_si128(first.as_ptr().add(2));
    let mut x3 = _mm_load_si128(first.as_ptr().add(3));
    let mut x4 = _mm_load_si128(first.as_ptr().add(4));
    let mut x5 = _mm_load_si128(first.as_ptr().add(5));
    let mut x6 = _mm_load_si128(first.as_ptr().add(6));
    let mut x7 = _mm_load_si128(first.as_ptr().add(7));

    // xor the initial CRC value
    x0 = _mm_xor_si128(x0, state);

    // all K_nnn constants are computed by bit_reverse(x^nnn mod POLY).
    const K_1023: u64 = 0xd7d8_6b2a_f73d_e740;
    const K_1087: u64 = 0x8757_d71d_4fcc_1000;
    let coeff_128 = build_const(K_1087, K_1023);

    // perform 128-byte folding.
    for chunk in rest {
        x0 = fold(coeff_128, x0, _mm_load_si128(chunk.as_ptr()));
        x1 = fold(coeff_128, x1, _mm_load_si128(chunk.as_ptr().add(1)));
        x2 = fold(coeff_128, x2, _mm_load_si128(chunk.as_ptr().add(2)));
        x3 = fold(coeff_128, x3, _mm_load_si128(chunk.as_ptr().add(3)));
        x4 = fold(coeff_128, x4, _mm_load_si128(chunk.as_ptr().add(4)));
        x5 = fold(coeff_128, x5, _mm_load_si128(chunk.as_ptr().add(5)));
        x6 = fold(coeff_128, x6, _mm_load_si128(chunk.as_ptr().add(6)));
        x7 = fold(coeff_128, x7, _mm_load_si128(chunk.as_ptr().add(7)));
    }

    // fold by distance of 112 bytes
    const K_895: u64 = 0x9478_74de_5950_52cb;
    const K_959: u64 = 0x9e73_5cb5_9b47_24da;
    x7 = fold(build_const(K_959, K_895), x0, x7);

    // fold by distance of 96 bytes
    const K_767: u64 = 0xe4ce_2cd5_5fea_0037;
    const K_831: u64 = 0x2fe3_fd29_20ce_82ec;
    x7 = fold(build_const(K_831, K_767), x1, x7);

    // fold by distance of 80 bytes
    const K_639: u64 = 0x0e31_d519_421a_63a5;
    const K_703: u64 = 0x2e30_2032_12ca_c325;
    x7 = fold(build_const(K_703, K_639), x2, x7);

    // fold by distance of 64 bytes
    const K_511: u64 = 0x081f_6054_a784_2df4;
    const K_575: u64 = 0x6ae3_efbb_9dd4_41f3;
    x7 = fold(build_const(K_575, K_511), x3, x7);

    // fold by distance of 48 bytes
    const K_383: u64 = 0x69a3_5d91_c373_0254;
    const K_447: u64 = 0xb5ea_1af9_c013_aca4;
    x7 = fold(build_const(K_447, K_383), x4, x7);

    // fold by distance of 32 bytes
    const K_255: u64 = 0x3be6_53a3_0fe1_af51;
    const K_319: u64 = 0x6009_5b00_8a9e_fa44;
    x7 = fold(build_const(K_319, K_255), x5, x7);

    // fold by distance of 16 bytes
    const K_127: u64 = 0xdabe_95af_c787_5f40; // same as table::TABLE_7[1]
    const K_191: u64 = 0xe05d_d497_ca39_3ae4; // same as table::TABLE_15[1]
    x7 = fold(build_const(K_191, K_127), x6, x7);

    // finally fold 16 bytes into 8 bytes.
    let r = _mm_clmulepi64_si128(x7, build_const(0, K_127), 0x00);
    let r = _mm_xor_si128(r, _mm_srli_si128(x7, 8));

    // barrett reduction.
    const MU: u64 = 0x9c3e_466c_1729_63d5;
    const POLY: u64 = 0x92d8_af2b_af0e_1e85;
    let polymu = build_const(POLY, MU);
    let t1 = _mm_clmulepi64_si128(r, polymu, 0x00);
    let t2 = _mm_clmulepi64_si128(t1, polymu, 0x10);
    let res = _mm_xor_si128(_mm_xor_si128(t2, _mm_slli_si128(t1, 8)), r);

    _mm_extract_epi64(res, 1) as u64
}
