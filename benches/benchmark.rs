// Copyright 2019 TiKV Project Authors. Licensed under MIT or Apache-2.0.

use crc::{Crc, CRC_64_XZ};
use criterion::*;
use rand::{thread_rng, RngCore};

const CRC: Crc<u64> = Crc::<u64>::new(&CRC_64_XZ);

fn bench_crc(c: &mut Criterion) {
    let mut group = c.benchmark_group("CRC64");
    let mut rng = thread_rng();

    for &size in &[8, 12, 16] {
        let mut buf = vec![0u8; 3 << size];
        rng.fill_bytes(&mut buf);

        group.throughput(Throughput::Bytes(3 << size));
        group.bench_with_input(BenchmarkId::new("crc::crc64", size), &buf, |b, buf| {
            b.iter(|| {
                let mut digest = CRC.digest();
                digest.update(&buf[..(1 << size)]);
                digest.update(&buf[(1 << size)..(2 << size)]);
                digest.update(&buf[(2 << size)..]);
                digest.finalize()
            })
        });
        group.bench_with_input(BenchmarkId::new("crc64fast::simd", size), &buf, |b, buf| {
            b.iter(|| {
                let mut digest = crc64fast::Digest::new();
                digest.write(&buf[..(1 << size)]);
                digest.write(&buf[(1 << size)..(2 << size)]);
                digest.write(&buf[(2 << size)..]);
                digest.sum64()
            })
        });
        group.bench_with_input(
            BenchmarkId::new("crc64fast::table", size),
            &buf,
            |b, buf| {
                b.iter(|| {
                    let mut digest = crc64fast::Digest::new_table();
                    digest.write(&buf[..(1 << size)]);
                    digest.write(&buf[(1 << size)..(2 << size)]);
                    digest.write(&buf[(2 << size)..]);
                    digest.sum64()
                })
            },
        );
    }
}

criterion_group!(benches, bench_crc);
criterion_main!(benches);
