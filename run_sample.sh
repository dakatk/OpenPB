RUST_BACKTRACE=1 cargo run --release -- \
    -d sample_data.json \
    -n sample_network.json \
    -t 4 \
    -e 10000 \
    -b 3