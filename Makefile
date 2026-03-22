CARGO := $(HOME)/.cargo/bin/cargo

.PHONY: build bench clean

build:
	cd ane_kernel && $(CARGO) build --release

bench: build
	cd ane_kernel && $(CARGO) run --release --example distilbert_bench

clean:
	cd ane_kernel && $(CARGO) clean
