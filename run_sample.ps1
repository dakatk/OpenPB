# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:RUST_BACKTRACE=1; cargo run --release -- -d sample_data.json -n sample_network.json