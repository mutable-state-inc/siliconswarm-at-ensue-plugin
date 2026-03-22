CARGO := $(HOME)/.cargo/bin/cargo

.PHONY: build bench chat clean

build:
	cd ane_kernel && $(CARGO) build --release

bench: build
	cd ane_kernel && $(CARGO) run --release --example bench

chat: build
	cd ane_kernel && $(CARGO) run --release --example phi_ane

chat-smollm: build
	cd ane_kernel && $(CARGO) run --release --example smollm_ane

.PHONY: build bench chat clean

clean:
	cd ane_kernel && $(CARGO) clean
