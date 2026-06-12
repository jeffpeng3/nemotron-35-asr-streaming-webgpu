all:
	wasm-pack build --target web

test:
	cargo test --target wasm32-unknown-unknown