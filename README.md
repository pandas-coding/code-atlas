# 🗺️ ChatVCode
*A lightning-fast, local-first, semantic-aware AI code assistant.*

**ChatVCode** is a privacy-first AI coding agent that understands your codebase. Unlike cloud-based assistants, ChatVCode runs entirely on your local machine, utilizing native system programming to provide instant codebase navigation and accurate AI assistance without memory bloat.

## ✨ Why ChatVCode?
* **Local-First & Private:** Your codebase never leaves your machine. 
* **Zero-Configuration:** Packaged as a single-binary. No Docker, no Python environment, no complex API setups.
* **Semantic-Aware RAG:** Integrates `tree-sitter` for AST parsing rather than blunt text-chunking, ensuring the LLM receives highly contextualized code chunks.
* **Peak Performance:** Built with Rust and C++ to utilize multi-core CPU/GPU acceleration, resulting in **10x faster indexing speeds** compared to Python-based implementations.

## 🏗️ Architecture Overview
ChatVCode is designed as a modular system to ensure high maintainability and performance.

* **`agent-llm`**: C++ native inference engine utilizing `llama.cpp` for local LLM integration.
* **`agent-parser`**: AST-based code chunker using `tree-sitter` for logic-preserving code analysis.
* **`agent-vdb`**: Local embedded vector database and `ort` (ONNX Runtime) for local vector embeddings.
* **`agent-core`**: Agent orchestration layer featuring high-performance multi-threaded scanning.
* **`agent-cli`**: The user interface and entry-point, powered by `ratatui` for a rich terminal experience.

*(Modular workspace architecture allowing interchangeable backends and easy unit-testing)*

## 🚀 Roadmap
- [x] **M1: Core Engine** - Multi-threaded file system traverse and `tree-sitter` AST chunking integration.
- [x] **M2: Semantic Search** - Local ONNX embedding integration and embedded Vector DB implementation.
- [x] **M3: Inference** - `llama.cpp` FFI implementation and streaming generation.
- [x] **M4: Agentic Brain** - Prompt-state machine for multi-step codebase reasoning.
- [ ] **M5: LSP Server** - `tower-lsp` implementation to serve directly into VS Code.

## 🛠️ Prerequisites
Before building, ensure you have:
* [Rust toolchain](https://rustup.rs/) (latest stable)
* [CMake](https://cmake.org/) (for compiling `llama.cpp`)
* A C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
* [GGUF Model](https://huggingface.co/models) (Place your coding model inside `~/.chatvcode/models/`)

## 📦 Getting Started
```bash
# Clone the repository
git clone --recursive https://github.com/YOUR_USERNAME/chatvcode.git
cd chatvcode

# Build the project
cargo build --release

# Initialize index for your current repository
./target/release/chatvcode index ./

# Ask a question about your codebase!
./target/release/chatvcode chat "Explain how the authentication middleware is implemented?"
```

## 🤖 Model Preparation

ChatVCode requires a **GGUF** format model for local inference. Place your model file in the default directory:

```
~/.chatvcode/models/
```

### Recommended Models

| Model | Size | Use Case |
|-------|------|----------|
| Qwen2.5-Coder-7B-Instruct | ~4.4 GB (Q4_K_M) | Best for coding tasks |
| DeepSeek-Coder-6.7B-Instruct | ~3.8 GB (Q4_K_M) | Good coding alternative |
| CodeLlama-7B-Instruct | ~3.8 GB (Q4_K_M) | Meta's coding model |

### Downloading a Model

```bash
# Create the models directory
mkdir -p ~/.chatvcode/models

# Download from Hugging Face (example: Qwen2.5-Coder-7B-Instruct Q4_K_M)
# Visit https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
# Download the .gguf file to ~/.chatvcode/models/
```

### Common Model Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No GGUF model found` | Empty or missing models directory | Download a `.gguf` file to `~/.chatvcode/models/` |
| `Invalid GGUF magic bytes` | Corrupt or non-GGUF file | Re-download the model file |
| `Unsupported GGUF version` | Model uses newer GGUF format | Update `chatvcode` to latest version |
| `Out of memory` | Model too large for available RAM | Use a smaller quantization (e.g., Q4_0 instead of Q8_0) |

## 💬 Chat Command Usage

### Basic Usage

```bash
# Ask a question about the current directory
chatvcode chat "What does the main function do?"

# Ask about a specific project
chatvcode chat "Explain the authentication flow" --path /path/to/project

# Use a specific model
chatvcode chat "How does routing work?" --model /path/to/model.gguf
```

### Generation Parameters

```bash
# Control output length
chatvcode chat "Explain this code" --max-tokens 1024

# Adjust creativity (lower = more deterministic)
chatvcode chat "Write a function" --temperature 0.3

# Limit sampling options
chatvcode chat "Refactor this" --top-k 50

# Override chat template
chatvcode chat "Question" --template chatml
```

### Output Modes

```bash
# Default: streaming output (tokens appear as generated)
chatvcode chat "Explain the codebase structure"

# Non-streaming: wait for complete response
chatvcode chat "Quick answer" --no-stream

# JSON output for programmatic use
chatvcode chat "List all API endpoints" --json
```

### System Prompt

```bash
# Custom system prompt
chatvcode chat "Review this code" --system-prompt "You are a senior Rust developer."
```

## 🤖 Agent Command Usage

`chatvcode agent` runs an **agentic loop** that autonomously explores your codebase through multi-step reasoning. Unlike `chat` (single-shot RAG), the agent decides which tools to call (search, read, list, grep), observes results, and iterates until it can answer.

### Prerequisites

The agent works best after indexing (so `search_code` / `search_symbol` are available). File-level tools (`list_files`, `read_file`, `grep_code`, `get_file_structure`) work without an index.

```bash
chatvcode index ./ --embedding-model <path/to/embedding.gguf>
```

### Basic Usage

```bash
# Ask an autonomous question about the project (defaults to current directory)
chatvcode agent "How is authentication implemented?" --path .

# Use a specific model and limit the number of steps
chatvcode agent "Find all API endpoints" --model /path/to/model.gguf --max-steps 15

# Print every step (thinking, tool calls, results)
chatvcode agent "Explain the error handling flow" --verbose

# Emit the final response as JSON (for programmatic consumption)
chatvcode agent "List the public types in src/lib.rs" --json
```

### Interactive Mode

```bash
# Multi-turn REPL with control commands
chatvcode agent -i --path .

# Inside the REPL:
#   /help     show available commands
#   /quit     exit
#   /retry    re-run the last question
#   /tools    list available tools
#   /verbose  toggle verbose output
#   /budget   show token budget usage
```

### Available Built-in Tools

| Tool | Description | Requires index? |
|------|-------------|-----------------|
| `search_code` | Semantic code search via embeddings | Yes |
| `search_symbol` | Look up symbols (functions/structs/...) in the metadata store | Yes |
| `read_file` | Read a file (with optional line range) | No |
| `list_files` | List files recursively (glob + depth filter) | No |
| `grep_code` | Regex search across files | No |
| `get_file_structure` | AST-based file structure overview | No |

### Output Modes

- **Default**: streams the final answer; prints tool call summaries and a stats footer.
- **`--verbose` / `-v`**: prints every thinking step, tool call, and result.
- **`--json`**: emits the final `AgentResponse` as JSON.
- **`--no-stream`**: waits for completion before printing.

### Common Options

| Option | Description |
|--------|-------------|
| `--path / -p` | Project directory to analyze (default: `.`) |
| `--model / -m` | GGUF model path (auto-discovered if omitted) |
| `--max-steps` | Maximum agent steps (default: 10) |
| `--timeout` | Agent timeout in seconds (default: 120) |
| `--tools` | Comma-separated allowlist of tool names |
| `--interactive / -i` | Interactive REPL mode |
| `--json` | JSON output mode |
| `--no-stream` | Disable streaming output |
| `--confirm-plan` | Require plan confirmation before execution |
| `--temperature` | Generation temperature (default: 0.7) |
| `--max-tokens` | Maximum generated tokens (default: 2048) |
| `--n-ctx` | Context window size (default: 8192) |
| `--n-gpu-layers` | GPU layers to offload (-1 for all) |
| `--embedding-model` | Embedding model for `search_code` (falls back to file tools if absent) |

## 🔧 Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| `CMake not found` | Install CMake and ensure it's in PATH |
| `link.exe failed` (Windows) | Ensure MSVC and CMake are installed; try `cargo clean` then rebuild |
| `llama.cpp source not found` | Run `git submodule update --init --recursive` |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| `Vector store not found` | Run `chatvcode index ./` first to build the index |
| `Context overflow` | Reduce `--max-tokens` or `--context-token-budget`, or increase `--n-ctx` |
| `No relevant code found` | Ensure the project has been indexed; check that source files are not in ignored directories |
| Slow inference | Use a smaller model quantization (Q4_0); enable GPU offload with `--n-gpu-layers -1` |

## 📂 Directory Structure
```text
chatvcode/
├── third_party/          # C/C++ dependencies (llama.cpp, tree-sitter, etc.)
├── crates/               # Rust modular workspace
│   ├── chatvcode-cli/        # CLI & TUI entry point
│   ├── chatvcode-agent/      # Agentic tool-use loop & state machine
│   ├── chatvcode-core/       # RAG orchestration & indexing pipeline
│   ├── chatvcode-parser/     # AST Code chunking engine
│   ├── chatvcode-vdb/        # Local Vector DB & ONNX Embeddings
│   └── chatvcode-llm/        # FFI LLM Inference wrapper
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
