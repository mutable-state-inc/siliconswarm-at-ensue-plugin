CARGO := $(HOME)/.cargo/bin/cargo

.PHONY: build bench verify clean

build:
	cd ane_kernel && $(CARGO) build --release

bench: build
	cd ane_kernel && $(CARGO) run --release --example distilbert_bench

verify: build
	cd ane_kernel && $(CARGO) run --release --example distilbert_verify

clean:
	cd ane_kernel && $(CARGO) clean
