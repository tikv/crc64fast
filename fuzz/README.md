# Fuzz Testing

This is the fuzz testing target for the `crc64fast` crate.

1. Install `afl` via `cargo install afl`
2. Build the fuzz target via `cargo afl build`
3. Generate a random input file via `head -c 1000 </dev/urandom >in/random`
4. Run the fuzz test via `cargo afl fuzz -i in -o out -- ./target/debug/fuzz`
