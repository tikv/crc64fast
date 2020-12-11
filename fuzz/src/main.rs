#[macro_use]
extern crate afl;
extern crate crc64fast;

use crc64fast::Digest;

fn main() {
    let digest_init = Digest::new();
    fuzz!(|data: &[u8]| {
        let mut digest = digest_init.clone();
        digest.write(data);
        digest.sum64();
    });
}
