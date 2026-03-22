CARGO := $(HOME)/.cargo/bin/cargo

.PHONY: build bench verify fmt lint clean

build:
	cd ane_kernel && $(CARGO) build --release --workspace

bench: build
	cd ane_kernel && $(CARGO) run --release --example distilbert_bench

verify: build
	cd ane_kernel && $(CARGO) run --release --example distilbert_verify

fmt:
	cd ane_kernel && $(CARGO) fmt --all

lint:
	cd ane_kernel && $(CARGO) clippy --release -p ane-bench -- -D warnings
	cd ane_kernel && $(CARGO) clippy --release --examples -p ane

clean:
	cd ane_kernel && $(CARGO) clean
