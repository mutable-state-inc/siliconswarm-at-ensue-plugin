CARGO := $(HOME)/.cargo/bin/cargo
PYTHON := $(if $(wildcard .venv/bin/python3),.venv/bin/python3,python3)
PIP := $(if $(wildcard .venv/bin/pip3),.venv/bin/pip3,pip3)

.PHONY: build bench verify fmt lint clean check-rust check-python download-models setup

check-rust:
	@if [ ! -f $(CARGO) ]; then \
		echo "Rust not found. Installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	else \
		echo "Rust OK"; \
	fi

check-python:
	@if [ ! -d .venv ]; then \
		echo "Creating Python venv..."; \
		python3 -m venv .venv; \
	fi
	@$(PYTHON) -c "import coremltools, numpy, huggingface_hub, transformers" 2>/dev/null || \
		(echo "Installing Python dependencies..." && $(PIP) install coremltools numpy huggingface_hub transformers)
	@echo "Python OK ($(PYTHON))"

download-models: check-python
	@echo "Downloading models..."
	@$(PYTHON) -c "import os, shutil; from huggingface_hub import snapshot_download; snap = snapshot_download('apple/ane-distilbert-base-uncased-finetuned-sst-2-english'); dst = '/tmp/DistilBERT_fp16.mlpackage'; os.path.exists(dst) or shutil.copytree(os.path.join(snap, 'DistilBERT_fp16.mlpackage'), dst)"
	@$(PYTHON) -c "from huggingface_hub import hf_hub_download; hf_hub_download('distilbert-base-uncased-finetuned-sst-2-english', 'model.safetensors'); hf_hub_download('distilbert-base-uncased', 'tokenizer.json')"
	@echo "Models OK"

setup: check-rust build check-python download-models
	@cd ane_kernel && $(CARGO) run --release -p ane-bench -- chip
	@cd ane_kernel && $(CARGO) run --release -p ane-bench -- ram

build: check-rust
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

bench-coreml: check-python
	$(PYTHON) benchmark_coreml.py

clean:
	cd ane_kernel && $(CARGO) clean
