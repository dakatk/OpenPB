# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:RUST_BACKTRACE=1; cargo run --release -- -d data.json -n network.json