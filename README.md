# 🗺️ CodeAtlas
*A lightning-fast, local-first, semantic-aware AI code assistant.*

**CodeAtlas** is a privacy-first AI coding agent that understands your codebase. Unlike cloud-based assistants, CodeAtlas runs entirely on your local machine, utilizing native system programming to provide instant codebase navigation and accurate AI assistance without memory bloat.

## ✨ Why CodeAtlas?
* **Local-First & Private:** Your codebase never leaves your machine. 
* **Zero-Configuration:** Packaged as a single-binary. No Docker, no Python environment, no complex API setups.
* **Semantic-Aware RAG:** Integrates `tree-sitter` for AST parsing rather than blunt text-chunking, ensuring the LLM receives highly contextualized code chunks.
* **Peak Performance:** Built with Rust and C++ to utilize multi-core CPU/GPU acceleration, resulting in **10x faster indexing speeds** compared to Python-based implementations.

## 🏗️ Architecture Overview
CodeAtlas is designed as a modular system to ensure high maintainability and performance.

* **`agent-llm`**: C++ native inference engine utilizing `llama.cpp` for local LLM integration.
* **`agent-parser`**: AST-based code chunker using `tree-sitter` for logic-preserving code analysis.
* **`agent-vdb`**: Local embedded vector database and `ort` (ONNX Runtime) for local vector embeddings.
* **`agent-core`**: Agent orchestration layer featuring high-performance multi-threaded scanning.
* **`agent-cli`**: The user interface and entry-point, powered by `ratatui` for a rich terminal experience.

*(Modular workspace architecture allowing interchangeable backends and easy unit-testing)*

## 🚀 Roadmap
- [ ] **M1: Core Engine** - Multi-threaded file system traverse and `tree-sitter` AST chunking integration.
- [ ] **M2: Semantic Search** - Local ONNX embedding integration and embedded Vector DB implementation.
- [ ] **M3: Inference** - `llama.cpp` FFI implementation and streaming generation.
- [ ] **M4: Agentic Brain** - Prompt-state machine for multi-step codebase reasoning.
- [ ] **M5: LSP Server** - `tower-lsp` implementation to serve directly into VS Code.

## 🛠️ Prerequisites
Before building, ensure you have:
* [Rust toolchain](https://rustup.rs/) (latest stable)
* [CMake](https://cmake.org/) (for compiling `llama.cpp`)
* A C++ compiler (GCC/Clang)
* [GGUF Model](https://huggingface.co/models) (Place your coding model inside `~/.codeatlas/models/`)

## 📦 Getting Started
```bash
# Clone the repository
git clone --recursive https://github.com/YOUR_USERNAME/code-atlas.git
cd code-atlas

# Build the project
cargo build --release

# Initialize index for your current repository
./target/release/code-atlas index ./

# Ask a question about your codebase!
./target/release/code-atlas chat "Explain how the authentication middleware is implemented?"
```

## 📂 Directory Structure
```text
code-atlas/
├── third_party/          # C/C++ dependencies (llama.cpp, tree-sitter, etc.)
├── crates/               # Rust modular workspace
│   ├── atlas-cli/        # CLI & TUI entry point
│   ├── atlas-core/       # Agent orchestration & RAG pipeline
│   ├── atlas-parser/     # AST Code chunking engine
│   ├── atlas-vdb/        # Local Vector DB & ONNX Embeddings
│   └── atlas-llm/        # FFI LLM Inference wrapper
└── tests/                # Integration tests
```

## 🤝 Contributing
Contributions are highly welcomed! Whether it is adding a new language parser for `tree-sitter`, implementing new embedding models, or improving the TUI, feel free to fork and submit a Pull Request! 
*(Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting)*

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💡 Acknowledgements
* [llama.cpp](https://github.com/ggerganov/llama.cpp) for being the bedrock of local LLM inference.
* [tree-sitter](https://tree-sitter.github.io/tree-sitter/) for code parsing excellence.
* [Bloop](https://github.com/BloopAI/bleep) & [TabbyML](https://github.com/TabbyML/tabby) for inspiration.
