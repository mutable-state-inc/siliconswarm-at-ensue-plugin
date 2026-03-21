CARGO := $(HOME)/.cargo/bin/cargo
GOFLAGS := -tags=ane_appleneuralengine
BENCH_DIR := $(shell go env GOPATH)/src/github.com/tmc/autoresearch-mlx-go-ane

.PHONY: ane sync build bench clean

ane:
	cd ane_kernel && $(CARGO) build --release

sync: ane
	mkdir -p $(BENCH_DIR)/ane_kernel/target/release
	cp ane_kernel/target/release/libane.dylib $(BENCH_DIR)/ane_kernel/target/release/libane.dylib
	cp ane_kernel/crates/ane/src/ffi.rs $(BENCH_DIR)/ane_kernel/crates/ane/src/ffi.rs

build: sync
	cd $(BENCH_DIR) && GOFLAGS="$(GOFLAGS)" go test -c -o /dev/null .

bench: build
	cd $(BENCH_DIR) && GOFLAGS="$(GOFLAGS)" go test -bench=BenchmarkGenerate -benchtime=1x -count=3 -run='^$$' -timeout=10m

clean:
	cd ane_kernel && $(CARGO) clean
