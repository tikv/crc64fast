crc64fast
=========

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
* an SSE/PCLMULQDQ-based implementation on modern x86/x86_64 processors.

| Algorithm             | Throughput (x86_64) |
|:----------------------|--------------------:|
| [crc 1.8.1]           |  0.5 GiB/s          |
| crc64fast (table)     |  2.3 GiB/s          |
| crc64fast (pclmulqdq) | 28.2 GiB/s          |

[crc 1.8.1]: https://crates.io/crates/crc

## TODO

This crate is mainly intended for use in TiKV only.
Features beyond AArch64 are unlikely to be implemented.

* [ ] AArch64 support based on PMULL
* [ ] `no_std` support
* [ ] Fuzz test
* [ ] Custom polynomial

## License

crc64fast is dual-licensed under

* Apache 2.0 license ([LICENSE-Apache] or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT] or <https://opensource.org/licenses/MIT>)
