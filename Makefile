CARGO := $(HOME)/.cargo/bin/cargo

.PHONY: build bench verify fmt lint clean check-rust check-python setup

check-rust:
	@if [ ! -f $(CARGO) ]; then \
		echo "Rust not found. Installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	else \
		echo "Rust OK"; \
	fi

check-python:
	@python3 -c "import coremltools, numpy, huggingface_hub" 2>/dev/null || \
		(echo "Installing Python dependencies..." && pip3 install coremltools numpy huggingface_hub)
	@echo "Python OK"

setup: check-rust check-python build
	@cd ane_kernel && $(CARGO) run --release -p ane-bench -- chip
	@cd ane_kernel && $(CARGO) run --release -p ane-bench -- ram

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

bench-coreml:
	python3 benchmark_coreml.py

clean:
	cd ane_kernel && $(CARGO) clean
