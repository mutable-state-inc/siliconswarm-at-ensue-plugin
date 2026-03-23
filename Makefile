CARGO := $(HOME)/.cargo/bin/cargo

.PHONY: build bench verify fmt lint clean check-rust check-python download-models setup

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

download-models: check-python
	@echo "Downloading models..."
	@python3 -c "import os, shutil; from huggingface_hub import snapshot_download; snap = snapshot_download('apple/ane-distilbert-base-uncased-finetuned-sst-2-english'); dst = '/tmp/DistilBERT_fp16.mlpackage'; os.path.exists(dst) or shutil.copytree(os.path.join(snap, 'DistilBERT_fp16.mlpackage'), dst)"
	@python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('distilbert-base-uncased-finetuned-sst-2-english', 'model.safetensors'); hf_hub_download('distilbert-base-uncased', 'tokenizer.json')"
	@echo "Models OK"

setup: check-rust check-python build download-models
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
