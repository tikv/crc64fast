crc64fast
=========

[![Build status](https://github.com/tikv/crc64fast/workflows/Rust/badge.svg)](https://github.com/tikv/crc64fast/actions?query=workflow%3ARust)
[![Latest Version](https://img.shields.io/crates/v/crc64fast.svg)](https://crates.io/crates/crc64fast)
[![Documentation](https://img.shields.io/badge/api-rustdoc-blue.svg)](https://docs.rs/crc64fast)

SIMD-accelerated CRC-64-ECMA computation
(similar to [`crc32fast`](https://crates.io/crates/crc32fast)).

## Usage

```rust
use crc64fast::Digest;

let mut c = Digest::new();
c.write(b"hello ");
c.write(b"world!");
let checksum = c.sum64();
assert_eq!(checksum, 0x8483_c0fa_3260_7d61);
```

## Performance

`crc64fast` provides two fast implementations, and the most performance one will
be chosen based on CPU feature at runtime.

* a fast, platform-agnostic table-based implementation, processing 16 bytes at a time.
* a SIMD-carryless-multiplication based implementation on modern processors:
    * using PCLMULQDQ + SSE 4.1 on x86/x86_64
    * using PMULL + NEON on AArch64 (64-bit ARM)

| Algorithm         | Throughput (x86_64) | Throughput (aarch64) |
|:------------------|--------------------:|---------------------:|
| [crc 1.8.1]       |  0.5 GiB/s          |  0.3 GiB/s           |
| crc64fast (table) |  2.3 GiB/s          |  1.8 GiB/s           |
| crc64fast (simd)  | 28.2 GiB/s          | 20.0 GiB/s           |

[crc 1.8.1]: https://crates.io/crates/crc

> **Note:** Since Rust has not stabilized SIMD support on AArch64, you need a
> nightly compiler and enable the `pmull` feature to use the SIMD-based
> implementation:
>
> ```toml
> [dependencies]
> crc64fast = { version = "1.0", features = ["pmull"] }
> ```

## TODO

This crate is mainly intended for use in TiKV only.
Features beyond AArch64 are unlikely to be implemented.

* [x] AArch64 support based on PMULL
* [ ] `no_std` support
* [x] Fuzz test
* [ ] Custom polynomial

## License

crc64fast is dual-licensed under

* Apache 2.0 license ([LICENSE-Apache](./LICENSE-Apache) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](./LICENSE-MIT) or <https://opensource.org/licenses/MIT>)
