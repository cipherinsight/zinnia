# Environment of every command run in /home/zhantong/zinnia-eval-run (sourced by
# zinnia-src/eval/tools/remote.sh on the local machine). The container has no internet access:
# everything below points at copies brought from the local machine.
export RUN=/home/zhantong/zinnia-eval-run
# Keep temporary files off the container's nearly full root overlay.
export TMPDIR=$RUN/tmp
# Cargo: the crate caches in ~/.cargo were completed from the local machine; never go online
# (also set in $RUN/.cargo/config.toml for builds that strip CARGO_* variables, e.g. risc0-build).
export CARGO_NET_OFFLINE=true
# The Rust toolchain is pinned by $RUN/rust-toolchain.toml (1.96.1); the container's default
# `stable` (1.94.1) is left as it is.
# Go for SP1's native gnark (sp1-recursion-gnark-ffi), offline from a copied module cache.
export GOROOT=$RUN/opt/go
export GOPATH=$RUN/opt/gopath
export GOCACHE=$RUN/cache/go-build
export GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local GOTELEMETRY=off GOFLAGS=-mod=mod
# libclang for bindgen (gnark FFI and RISC Zero build scripts); the container has no clang.
export LIBCLANG_PATH=$RUN/opt/libclang
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
# Python tooling caches.
export UV_CACHE_DIR=$RUN/cache/uv UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python UV_OFFLINE=1
export PIP_NO_INDEX=1
# The harness's step memory cap: no systemd and a read-only cgroup tree in this container, so
# the cap is enforced by the harness's RSS watchdog (benchmarking/harness/isolation.py).
export ZINNIA_HARNESS_MEMORY_CAP=watchdog
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.sp1/bin:$HOME/.nargo/bin:$HOME/.bb:$GOROOT/bin:$PATH
# protoc for prost-build (sp1-prover-types), the release binary of the local machine's version.
export PROTOC=$RUN/opt/protoc/bin/protoc PROTOC_INCLUDE=$RUN/opt/protoc/include
# The RISC Zero recursion circuit (risc0-circuit-recursion downloads it at build time; its
# build script checks the SHA-256 744b999f...), copied from the local build.
export RECURSION_SRC_PATH=$RUN/opt/risc0/recursion_zkr.zip
