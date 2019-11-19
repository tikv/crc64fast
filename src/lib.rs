// Copyright 2019 TiKV Project Authors. Licensed under MIT or Apache-2.0.

//! `crc64fast`
//! ===========
//!
//! SIMD-accelerated CRC-64-ECMA computation
//! (similar to [`crc32fast`](https://crates.io/crates/crc32fast)).
//!
//! ## Usage
//!
//! ```
//! use crc64fast::Digest;
//!
//! let mut c = Digest::new();
//! c.write(b"hello ");
//! c.write(b"world!");
//! let checksum = c.sum64();
//! assert_eq!(checksum, 0x8483_c0fa_3260_7d61);
//! ```

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod pclmulqdq;
mod table;

type UpdateFn = fn(u64, &[u8]) -> u64;

/// Represents an in-progress CRC-64 computation.
#[derive(Clone)]
pub struct Digest {
    computer: UpdateFn,
    state: u64,
}

impl Digest {
    /// Creates a new `Digest`.
    ///
    /// It will perform runtime CPU feature detection to determine which
    /// algorithm to choose.
    pub fn new() -> Self {
        Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            computer: pclmulqdq::get_update(),
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            computer: table::update,
            state: !0,
        }
    }

    /// Creates a new `Digest` using table-based algorithm.
    pub fn new_table() -> Self {
        Self {
            computer: table::update,
            state: !0,
        }
    }

    /// Writes some data into the digest.
    pub fn write(&mut self, bytes: &[u8]) {
        self.state = (self.computer)(self.state, bytes);
    }

    /// Computes the current CRC-64-ECMA value.
    pub fn sum64(&self) -> u64 {
        !self.state
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Digest;
    use crc::crc64::checksum_ecma;
    use proptest::collection::size_range;
    use proptest::prelude::*;

    #[test]
    fn test_standard_vectors() {
        static CASES: &[(&[u8], u64)] = &[
            (b"", 0),
            (b"@", 0x7b1b_8ab9_8fa4_b8f8),
            (b"1\x97", 0xfeb8_f7a1_ae3b_9bd4),
            (b"M\"\xdf", 0xc016_0ce8_dd46_74d3),
            (b"l\xcd\x13\xd7", 0x5c60_a6af_8299_6ea8),

            (&[0; 32], 0xc95a_f861_7cd5_330c),
            (&[255; 32], 0xe95d_ce9e_faa0_9acf),
            (b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F", 0x7fe5_71a5_8708_4d10),

            (&[0; 1024], 0xc378_6397_2069_270c),
        ];

        for (input, result) in CASES {
            let mut hasher = Digest::new();
            hasher.write(input);
            assert_eq!(hasher.sum64(), *result, "test case {:x?}", input);
        }
    }

    fn any_buffer() -> <Box<[u8]> as Arbitrary>::Strategy {
        any_with::<Box<[u8]>>(size_range(..65536).lift())
    }

    prop_compose! {
        fn bytes_and_split_index()
            (bytes in any_buffer())
            (index in 0..=bytes.len(), bytes in Just(bytes)) -> (Box<[u8]>, usize)
        {
            (bytes, index)
        }
    }

    proptest! {
        #[test]
        fn equivalent_to_crc(bytes in any_buffer()) {
            let mut hasher = Digest::new();
            hasher.write(&bytes);
            prop_assert_eq!(hasher.sum64(), checksum_ecma(&bytes));
        }

        #[test]
        fn concatenation((bytes, split_index) in bytes_and_split_index()) {
            let mut hasher_1 = Digest::new();
            hasher_1.write(&bytes);
            let mut hasher_2 = Digest::new();
            let (left, right) = bytes.split_at(split_index);
            hasher_2.write(left);
            hasher_2.write(right);
            prop_assert_eq!(hasher_1.sum64(), hasher_2.sum64());
        }

        #[test]
        fn state_cloning(left in any_buffer(), right in any_buffer()) {
            let mut hasher_1 = Digest::new();
            hasher_1.write(&left);
            let mut hasher_2 = hasher_1.clone();
            hasher_1.write(&right);
            hasher_2.write(&right);
            prop_assert_eq!(hasher_1.sum64(), hasher_2.sum64());
        }
    }
}
