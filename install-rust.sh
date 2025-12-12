#!/bin/bash
set -e

export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"

# Check if Rust is properly installed and configured
if ! rustc --version &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    source $HOME/.cargo/env
fi

# Ensure default toolchain is set
rustup default stable 2>/dev/null || true

# Install wasm-pack if not present
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Verify installations
echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"
echo "wasm-pack version: $(wasm-pack --version)"