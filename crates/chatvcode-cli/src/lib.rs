use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub mod agent;

use chatvcode_core::{
    ChatOptions, ChatResponse, ChatVCodeError, EmbeddingOptions, ErrorSeverity, IndexOptions,
    IndexResult, SearchOptions, SourceReference, chat_with_context, chat_with_context_stream,
    index_path_with_options, search, search_with_service,
};
use chatvcode_llm::{
    ChatPromptBuilder, ChatSession, ChatTemplate, ChatvcodeConfig, GenerationParams,
    LlamaEmbeddingService, LlamaModel, LlamaService, LlmConfig, LlmService, MockLlmService,
    StreamEvent, list_models, estimate_memory, recommend_gpu_layers, format_bytes,
};
use chatvcode_parser::parse_source;
use chatvcode_vdb::{EmbeddingConfig, EmbeddingService};
use clap::{Parser, Subcommand};
use rustyline::DefaultEditor;

pub use chatvcode_core;
pub use chatvcode_llm;
pub use chatvcode_parser;
pub use chatvcode_vdb;

// ---------------------------------------------------------------------------
// GGUF Embedding Adapter
// ---------------------------------------------------------------------------

/// Adapter that wraps [`LlamaEmbeddingService`] to implement [`EmbeddingService`].
///
/// This allows the same GGUF model used for LLM inference to also generate
/// embeddings for RAG retrieval, eliminating the need for a separate ONNX model.
pub(crate) struct LlamaEmbeddingAdapter {
    inner: LlamaEmbeddingService,
}

impl LlamaEmbeddingAdapter {
    fn new(service: LlamaEmbeddingService) -> Self {
        Self { inner: service }
    }
}

impl EmbeddingService for LlamaEmbeddingAdapter {
    fn embed(&self, texts: &[&str]) -> chatvcode_vdb::VdbResult<Vec<Vec<f32>>> {
        self.inner
            .embed(texts)
            .map_err(|e| chatvcode_vdb::VdbError::inference(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

fn is_gguf_path(path: &str) -> bool {
    path.to_lowercase().ends_with(".gguf")
}

fn load_gguf_embedding_service(
    model_path: &std::path::Path,
    n_gpu_layers: i32,
    n_threads: i32,
    llm_verbose_log: bool,
) -> Result<Box<dyn EmbeddingService>, ChatVCodeError> {
    eprintln!(
        "🔍 Loading GGUF embedding model from {}...",
        model_path.display()
    );
    let embed_svc = LlamaEmbeddingService::from_path(
        model_path,
        512,
        n_threads,
        n_gpu_layers,
        llm_verbose_log,
    )
    .map_err(|e| {
        ChatVCodeError::internal(format!("Failed to load GGUF embedding model: {e}"))
            .with_severity(ErrorSeverity::Unrecoverable)
    })?;
    eprintln!(
        "✓ GGUF embedding model loaded (dim={}).",
        embed_svc.dimension()
    );
    Ok(Box::new(LlamaEmbeddingAdapter::new(embed_svc)) as Box<dyn EmbeddingService>)
}

/// CLI argument parser for the `chatvcode` tool.
#[derive(Parser)]
#[command(name = "chatvcode", version, about = "Code indexing, search, and AI chat tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Index source files in the given path and display statistics.
    Index {
        #[arg(help = "Path to the project directory or source file to index")]
        path: String,
        /// Path to the incremental state file (enables incremental indexing).
        #[arg(long, help = "Path to save/load incremental state")]
        state_file: Option<String>,
        /// Large-file threshold in bytes (default: 1048576 = 1MB).
        #[arg(long, default_value = "1048576", help = "Large file threshold in bytes")]
        large_file_threshold: usize,
        /// Maximum lines to read from large files (default: 500).
        #[arg(long, default_value = "500", help = "Max lines to read from large files")]
        large_file_max_lines: usize,
        /// Chunk split threshold in characters (default: 3000, 0 to disable).
        #[arg(long, default_value = "3000", help = "Max chars before splitting a chunk")]
        chunk_split_threshold: usize,
        /// Path to the embedding model file (GGUF or ONNX). Takes priority over --model.
        #[arg(long, help = "Path to the embedding model file (GGUF or ONNX)")]
        embedding_model: Option<String>,
        /// Path to the tokenizer JSON file (required for ONNX models).
        #[arg(long, help = "Path to the tokenizer JSON file for ONNX embeddings")]
        embedding_tokenizer: Option<String>,
        /// Embedding vector dimension (required if --embedding-model is set).
        #[arg(long, default_value = "0", help = "Embedding vector dimension")]
        embedding_dimension: usize,
        /// Maximum token length for embedding input (default: 512).
        #[arg(long, default_value = "512", help = "Max tokens for embedding input")]
        embedding_max_tokens: usize,
        /// Path to save the vector store file (default: <path>/.chatvcode/vectors.vdb).
        #[arg(long, help = "Path to save the vector store file")]
        vector_store_path: Option<String>,
        /// Embedding batch size (default: 32).
        #[arg(long, default_value = "32", help = "Embedding batch size")]
        embedding_batch_size: usize,
        /// Path to GGUF model file for embedding (auto-discovered from ~/.chatvcode/models/ if
        /// not set). Ignored if --embedding-model is set.
        #[arg(long, help = "Path to the GGUF model file for embedding")]
        model: Option<String>,
        /// Number of threads for GGUF embedding (default: auto).
        #[arg(long, help = "Number of threads for GGUF embedding")]
        n_threads: Option<i32>,
        /// Number of GPU layers to offload for GGUF embedding (default: 0, -1 = all).
        #[arg(long, default_value = "0", help = "Number of GPU layers for GGUF embedding")]
        n_gpu_layers: i32,
        /// Enable verbose llama.cpp/ggml log output (tensor creation, backend registration, etc.).
        #[arg(long, default_value_t = false, num_args = 0..=1, help = "Enable verbose llama.cpp/ggml logging (default: false)")]
        llm_verbose_log: bool,
    },
    /// Perform semantic search over indexed code chunks.
    Search {
        #[arg(help = "Search query text")]
        query: String,
        #[arg(long, help = "Path to the indexed project directory")]
        path: Option<String>,
        /// Path to the embedding model file (GGUF or ONNX).
        #[arg(long, help = "Path to the embedding model file (GGUF or ONNX)")]
        embedding_model: Option<String>,
        /// Path to the tokenizer JSON file (required for ONNX models).
        #[arg(long, help = "Path to the tokenizer JSON file for ONNX embeddings")]
        embedding_tokenizer: Option<String>,
        /// Embedding vector dimension (required if --embedding-model is set).
        #[arg(long, default_value = "0", help = "Embedding vector dimension")]
        embedding_dimension: usize,
        /// Maximum token length for embedding input (default: 512).
        #[arg(long, default_value = "512", help = "Max tokens for embedding input")]
        embedding_max_tokens: usize,
        /// Path to the vector store file (default: <path>/.chatvcode/vectors.vdb).
        #[arg(long, help = "Path to the vector store file")]
        vector_store_path: Option<String>,
        /// Number of top results to return (default: 10).
        #[arg(long, default_value = "10", help = "Number of top results to return")]
        top_k: usize,
        /// Minimum similarity score threshold (0.0 to 1.0).
        #[arg(long, help = "Minimum similarity score threshold")]
        min_score: Option<f32>,
    },
    /// Ask a question about a codebase using RAG-enhanced LLM inference.
    ///
    /// This command indexes the project (or uses an existing index), retrieves
    /// relevant code context via semantic search, and generates an answer using
    /// a local GGUF language model.
    ///
    /// Examples:
    ///   chatvcode chat "What does the main function do?" --path=./my-project
    ///   chatvcode chat "Explain error handling" --path=./my-project --model=./model.gguf
    ///   chatvcode chat "How is parsing done?" --path=./my-project --stream=false
    Chat {
        /// The question to ask about the codebase.
        question: String,
        /// Path to the project directory to analyze.
        #[arg(short, long, default_value = ".", help = "Path to the project directory")]
        path: String,
        /// Path to the GGUF model file (auto-discovered from ~/.chatvcode/models/ if not set).
        #[arg(short, long, help = "Path to the GGUF model file")]
        model: Option<String>,
        /// Temperature for generation (default: 0.7).
        #[arg(short, long, default_value = "0.7", help = "Temperature for generation")]
        temperature: f32,
        /// Maximum number of tokens to generate (default: 2048, range: 1-65536).
        #[arg(long, default_value = "2048", help = "Maximum number of tokens to generate")]
        max_tokens: i32,
        /// Top-k sampling parameter (default: 40).
        #[arg(long, default_value = "40", help = "Top-k sampling parameter")]
        top_k: i32,
        /// Top-p (nucleus) sampling parameter (default: 0.9).
        #[arg(long, default_value = "0.9", help = "Top-p sampling parameter")]
        top_p: f32,
        /// Chat template to use (auto, raw, chatml, llama3).
        #[arg(long, default_value = "auto", help = "Chat template")]
        template: String,
        /// System prompt for the LLM.
        #[arg(long, help = "System prompt for the LLM")]
        system_prompt: Option<String>,
        /// Enable streaming output (default: true, use --stream=false to disable).
        #[arg(long, default_value_t = true, num_args = 0..=1, help = "Enable streaming output (default: true, use --stream=false to disable)")]
        stream: bool,
        /// Output result as JSON.
        #[arg(long, help = "Output result as JSON")]
        json: bool,
        /// Context window size for the model (default: 8192).
        ///
        /// Larger values allow more RAG context but use more memory.
        /// For RAG mode with code snippets, 8192 is recommended.
        #[arg(long, default_value = "8192", help = "Context window size")]
        n_ctx: u32,
        /// Number of threads for inference (default: auto).
        #[arg(long, help = "Number of threads for inference")]
        n_threads: Option<i32>,
        /// Number of GPU layers to offload (default: 0, -1 = all).
        #[arg(long, default_value = "0", help = "Number of GPU layers (-1 for all)")]
        n_gpu_layers: i32,
        /// Path to the embedding model file (GGUF or ONNX). Takes priority over --model for embeddings.
        #[arg(long, help = "Path to the embedding model file (GGUF or ONNX)")]
        embedding_model: Option<String>,
        /// Path to the embedding tokenizer file (required for ONNX models).
        #[arg(long, help = "Path to the tokenizer JSON file for ONNX embeddings")]
        embedding_tokenizer: Option<String>,
        /// Embedding vector dimension (default: 0 = auto).
        #[arg(long, default_value = "0", help = "Embedding vector dimension")]
        embedding_dimension: usize,
        /// Maximum embedding token length (default: 512).
        #[arg(long, default_value = "512", help = "Max tokens for embedding input")]
        embedding_max_tokens: usize,
        /// Number of context snippets to retrieve (default: 8).
        #[arg(long, default_value = "16", help = "Number of context snippets to retrieve")]
        top_k_retrieval: usize,
        /// Minimum similarity score for retrieval (0.0-1.0).
        #[arg(long, help = "Minimum similarity score for retrieval")]
        min_score: Option<f32>,
        /// Context token budget (0 = unlimited).
        #[arg(long, default_value = "0", help = "Max tokens allocated to context (0=unlimited)")]
        context_token_budget: usize,
        /// Use mock LLM for testing (no real model needed).
        #[arg(long, hide = true, help = "Use mock LLM service for testing")]
        mock_llm: bool,
        /// Mock LLM response text (for testing with --mock-llm).
        #[arg(long, hide = true, help = "Response text for mock LLM")]
        mock_llm_response: Option<String>,
        /// Enable RAG retrieval (default: true, use --retrieval=false for LLM-only mode).
        /// When enabled, indexes the project and uses semantic search to provide context.
        /// When disabled (--retrieval=false), queries the LLM directly without code context.
        #[arg(long, default_value_t = true, num_args = 0..=1, help = "Enable RAG retrieval (default: true, use --retrieval=false to disable)")]
        retrieval: bool,
        /// Enable interactive multi-turn chat mode.
        ///
        /// When enabled, enters a REPL loop where you can ask multiple questions
        /// with conversation history preserved across turns. Use `/quit` to exit,
        /// `/clear` to clear history, `/help` for commands.
        #[arg(long, default_value_t = false, num_args = 0..=1, help = "Enable interactive multi-turn chat mode")]
        interactive: bool,
        /// Enable verbose llama.cpp/ggml log output (tensor creation, backend registration, etc.).
        #[arg(long, default_value_t = false, num_args = 0..=1, help = "Enable verbose llama.cpp/ggml logging (default: false)")]
        llm_verbose_log: bool,
        /// Path to a configuration file (JSON). Overrides: CLI > config file > defaults.
        #[arg(long, help = "Path to configuration file (~/.chatvcode/config.json)")]
        config: Option<String>,
    },
    /// Run an autonomous agent that explores the codebase using built-in tools.
    ///
    /// The agent uses a Thinking ↔ Acting state machine to perform multi-step
    /// reasoning, invokes tools (read_file, list_files, grep_code, etc.), and
    /// produces a final answer backed by an execution trace.
    ///
    /// Examples:
    ///   chatvcode agent "How is error handling structured?" --path ./my-project
    ///   chatvcode agent "Where is the entry point?" -i --mock-llm
    Agent(agent::AgentCommand),
    /// Manage models: list, inspect, estimate memory, and configure GPU offloading.
    ///
    /// Examples:
    ///   chatvcode model list                # List all discovered models
    ///   chatvcode model info <path>         # Show model metadata
    ///   chatvcode model memory <path>       # Estimate memory usage
    ///   chatvcode model gpu <path>          # Recommend GPU layer count
    ///   chatvcode model config show         # Show current configuration
    ///   chatvcode model config init         # Create default config file
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
}

/// Model management subcommands.
#[derive(Subcommand)]
pub enum ModelAction {
    /// List all discovered GGUF models in search directories.
    List,
    /// Show detailed metadata for a specific model.
    Info {
        /// Path to the GGUF model file.
        path: String,
    },
    /// Estimate memory usage for a model at a given context size.
    Memory {
        /// Path to the GGUF model file.
        path: String,
        /// Context window size (default: 8192).
        #[arg(long, default_value = "8192", help = "Context window size")]
        n_ctx: u32,
    },
    /// Recommend GPU layer offloading based on model size and VRAM.
    Gpu {
        /// Path to the GGUF model file.
        path: String,
        /// Available VRAM in GB (optional, uses heuristic if not specified).
        #[arg(long, help = "Available VRAM in GB")]
        vram_gb: Option<u64>,
    },
    /// Manage configuration file.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

/// Configuration file management subcommands.
#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show current configuration (merged from file + defaults).
    Show,
    /// Initialize a default configuration file.
    Init,
    /// Validate a configuration file.
    Validate {
        /// Path to config file (default: ~/.chatvcode/config.json).
        #[arg(long, help = "Path to config file")]
        path: Option<String>,
    },
}

/// Initializes the logger and runs the CLI.
///
/// Parses command-line arguments and dispatches to the appropriate handler.
/// Returns an error if the operation fails fatally.
pub fn run() -> Result<(), ChatVCodeError> {
    env_logger::init();
    let cli = Cli::parse();
    execute(cli)
}

/// Executes the given CLI command.
pub fn execute(cli: Cli) -> Result<(), ChatVCodeError> {
    match cli.command {
        Commands::Index {
            path,
            state_file,
            large_file_threshold,
            large_file_max_lines,
            chunk_split_threshold,
            embedding_model,
            embedding_tokenizer,
            embedding_dimension,
            embedding_max_tokens,
            vector_store_path,
            embedding_batch_size,
            model,
            n_threads,
            n_gpu_layers,
            llm_verbose_log,
        } => {
            // Resolve embedding strategy:
            //   --embedding-model (GGUF or ONNX) > --model (GGUF) > auto-discover (GGUF)
            let use_onnx = embedding_model
                .as_ref()
                .is_some_and(|p| !is_gguf_path(p));

            let embedding = if use_onnx {
                let model_path = embedding_model.as_ref().unwrap();
                let mut config =
                    EmbeddingConfig::new(model_path, embedding_dimension, embedding_max_tokens);
                if let Some(ref tokenizer) = embedding_tokenizer {
                    config = config.with_tokenizer_path(tokenizer);
                }
                let vstore_path = vector_store_path.as_deref().map_or_else(
                    || PathBuf::from(&path).join(".chatvcode").join("vectors.vdb"),
                    PathBuf::from,
                );
                Some(EmbeddingOptions {
                    config,
                    vector_store_path: vstore_path,
                    batch_size: embedding_batch_size,
                })
            } else {
                None
            };

            let options = IndexOptions {
                incremental_state_path: state_file.map(PathBuf::from),
                large_file_threshold,
                large_file_max_lines,
                chunk_split_threshold,
                embedding,
            };

            if use_onnx {
                let result = index_path_with_options(&path, &parse_source, &options)?;
                print_index_result(&result);
            } else {
                // GGUF path: --embedding-model (.gguf) > --model > auto-discover
                let gguf_model = embedding_model
                    .as_deref()
                    .or(model.as_deref());
                let result = index_with_gguf(
                    &path,
                    gguf_model,
                    &options,
                    embedding_batch_size,
                    vector_store_path.as_deref(),
                    n_threads,
                    n_gpu_layers,
                    llm_verbose_log,
                )?;
                print_index_result(&result);
            }
        }
        Commands::Search {
            query,
            path,
            embedding_model,
            embedding_tokenizer,
            embedding_dimension,
            embedding_max_tokens,
            vector_store_path,
            top_k,
            min_score,
        } => {
            let project_path = path.unwrap_or_else(|| ".".to_string());

            let model_path = match embedding_model {
                Some(p) => p,
                None => {
                    return Err(ChatVCodeError::invalid_input(
                        "--embedding-model is required for search",
                    ));
                }
            };

            let vstore_path = vector_store_path.map(PathBuf::from).unwrap_or_else(|| {
                PathBuf::from(&project_path)
                    .join(".chatvcode")
                    .join("vectors.vdb")
            });

            if is_gguf_path(&model_path) {
                let n_threads = num_cpus::get() as i32;
                let embed_svc = LlamaEmbeddingService::from_path(
                    &PathBuf::from(&model_path),
                    512,
                    n_threads,
                    0,
                    false,
                )
                .map_err(|e| {
                    ChatVCodeError::internal(format!(
                        "Failed to load GGUF embedding model: {e}"
                    ))
                    .with_severity(ErrorSeverity::Unrecoverable)
                })?;
                let adapter = LlamaEmbeddingAdapter::new(embed_svc);
                let config = EmbeddingConfig::new(&model_path, 0, embedding_max_tokens);
                let mut search_opts =
                    SearchOptions::new(config, vstore_path).with_top_k(top_k);
                if let Some(ms) = min_score {
                    search_opts = search_opts.with_min_score(ms);
                }
                let results = search_with_service(
                    &query,
                    &PathBuf::from(&project_path),
                    &parse_source,
                    &search_opts,
                    &adapter,
                )?;
                print_search_results(&query, &results);
            } else {
                let mut config =
                    EmbeddingConfig::new(&model_path, embedding_dimension, embedding_max_tokens);
                if let Some(tokenizer) = embedding_tokenizer {
                    config = config.with_tokenizer_path(&tokenizer);
                }

                let mut search_opts =
                    SearchOptions::new(config, vstore_path).with_top_k(top_k);
                if let Some(ms) = min_score {
                    search_opts = search_opts.with_min_score(ms);
                }

                let results = search(&query, &project_path, &parse_source, &search_opts)?;
                print_search_results(&query, &results);
            }
        }
        Commands::Chat {
            question,
            path,
            model,
            temperature,
            max_tokens,
            top_k,
            top_p,
            template,
            system_prompt,
            stream,
            json,
            n_ctx,
            n_threads,
            n_gpu_layers,
            embedding_model,
            embedding_tokenizer,
            embedding_dimension,
            embedding_max_tokens,
            top_k_retrieval,
            min_score,
            context_token_budget,
            mock_llm,
            mock_llm_response,
            retrieval,
            interactive,
            llm_verbose_log,
            config,
        } => {
            run_chat(ChatArgs {
                question,
                path,
                model,
                temperature,
                max_tokens,
                top_k,
                top_p,
                template,
                system_prompt,
                stream,
                json,
                n_ctx,
                n_threads,
                n_gpu_layers,
                embedding_model,
                embedding_tokenizer,
                embedding_dimension,
                embedding_max_tokens,
                top_k_retrieval,
                min_score,
                context_token_budget,
                mock_llm,
                mock_llm_response,
                retrieval,
                interactive,
                llm_verbose_log,
                config,
            })?;
        }
        Commands::Model { action } => {
            handle_model_command(action)?;
        }
        Commands::Agent(cmd) => {
            agent::run_agent(cmd)?;
        }
    }
    Ok(())
}

/// Handle model management commands.
fn handle_model_command(action: ModelAction) -> Result<(), ChatVCodeError> {
    match action {
        ModelAction::List => {
            let models = list_models();
            if models.is_empty() {
                eprintln!("No GGUF models found in search directories.");
                eprintln!("Search paths:");
                for (dir, source) in chatvcode_llm::model_search_dirs() {
                    eprintln!("  {} ({})", dir.display(), source);
                }
                return Ok(());
            }

            println!("Discovered {} model(s):\n", models.len());
            for (i, model) in models.iter().enumerate() {
                println!("[{}] {}", i + 1, model.summary_line());
                println!("    Path: {}", model.path.display());
                println!();
            }
        }
        ModelAction::Info { path } => {
            let model_path = PathBuf::from(&path);
            if !model_path.exists() {
                return Err(ChatVCodeError::invalid_input(format!(
                    "Model file not found: {}",
                    model_path.display()
                )));
            }

            let meta = chatvcode_llm::pre_validate_model(&model_path).map_err(|e| {
                ChatVCodeError::internal(format!("Failed to read model metadata: {e}"))
            })?;

            println!("{}", chatvcode_llm::format_gguf_summary(&model_path, &meta));
        }
        ModelAction::Memory { path, n_ctx } => {
            let model_path = PathBuf::from(&path);
            if !model_path.exists() {
                return Err(ChatVCodeError::invalid_input(format!(
                    "Model file not found: {}",
                    model_path.display()
                )));
            }

            let estimate = estimate_memory(&model_path, n_ctx).map_err(|e| {
                ChatVCodeError::internal(format!("Failed to estimate memory: {e}"))
            })?;

            println!("Memory Estimation for {}", model_path.display());
            println!("Context size: {n_ctx} tokens\n");
            println!("{}", estimate.summary());
        }
        ModelAction::Gpu { path, vram_gb } => {
            let model_path = PathBuf::from(&path);
            if !model_path.exists() {
                return Err(ChatVCodeError::invalid_input(format!(
                    "Model file not found: {}",
                    model_path.display()
                )));
            }

            let vram_bytes = vram_gb.map(|gb| gb * 1024 * 1024 * 1024);
            let recommendation = recommend_gpu_layers(&model_path, vram_bytes).map_err(|e| {
                ChatVCodeError::internal(format!("Failed to recommend GPU layers: {e}"))
            })?;

            println!("GPU Layer Recommendation for {}", model_path.display());
            if let Some(gb) = vram_gb {
                println!("Available VRAM: {gb} GB");
            } else {
                println!("Available VRAM: (using heuristic estimate)");
            }
            println!();
            println!("{}", recommendation.summary());
        }
        ModelAction::Config { action } => {
            handle_config_command(action)?;
        }
    }
    Ok(())
}

/// Handle configuration file management commands.
fn handle_config_command(action: ConfigAction) -> Result<(), ChatVCodeError> {
    match action {
        ConfigAction::Show => {
            let config = ChatvcodeConfig::load_default().unwrap_or_else(|e| {
                eprintln!("Warning: Could not load config file: {e}");
                eprintln!("Using default configuration.");
                ChatvcodeConfig::default()
            });

            let json = serde_json::to_string_pretty(&config).map_err(|e| {
                ChatVCodeError::internal(format!("Failed to serialize config: {e}"))
            })?;

            // Show which config files exist and their priority
            let local_path = chatvcode_llm::local_config_path();
            let global_path = chatvcode_llm::default_config_path();
            
            println!("Configuration file search paths (priority: high → low):");
            println!("  1. Local:  {} {}", 
                local_path.display(),
                if local_path.exists() { "✓ exists" } else { "✗ not found" }
            );
            println!("  2. Global: {} {}", 
                global_path.display(),
                if global_path.exists() { "✓ exists" } else { "✗ not found" }
            );
            println!("  3. Built-in defaults");
            println!();
            
            if local_path.exists() && global_path.exists() {
                println!("Merged configuration (local overrides global):");
            } else if local_path.exists() {
                println!("Configuration (from local file):");
            } else if global_path.exists() {
                println!("Configuration (from global file):");
            } else {
                println!("Configuration (built-in defaults):");
            }
            println!("{json}");
            
            println!();
            println!("Priority: CLI arguments > local config > global config > built-in defaults");
        }
        ConfigAction::Init => {
            let config_path = chatvcode_llm::default_config_path();
            if config_path.exists() {
                eprintln!("Configuration file already exists: {}", config_path.display());
                eprintln!("Use 'chatvcode model config show' to view current configuration.");
                return Ok(());
            }

            let config = ChatvcodeConfig::default();
            config.save_to(&config_path).map_err(|e| {
                ChatVCodeError::internal(format!("Failed to save config: {e}"))
            })?;

            println!("Created default configuration file: {}", config_path.display());
            println!("Edit this file to customize default settings.");
        }
        ConfigAction::Validate { path } => {
            let config_path = path
                .map(PathBuf::from)
                .unwrap_or_else(chatvcode_llm::default_config_path);

            if !config_path.exists() {
                return Err(ChatVCodeError::invalid_input(format!(
                    "Config file not found: {}",
                    config_path.display()
                )));
            }

            match ChatvcodeConfig::load_from(&config_path) {
                Ok(_) => {
                    println!("✓ Configuration file is valid: {}", config_path.display());
                }
                Err(e) => {
                    return Err(ChatVCodeError::invalid_input(format!(
                        "Configuration file validation failed: {e}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Arguments for the chat command, extracted from the CLI.
struct ChatArgs {
    question: String,
    path: String,
    model: Option<String>,
    temperature: f32,
    max_tokens: i32,
    top_k: i32,
    top_p: f32,
    template: String,
    system_prompt: Option<String>,
    stream: bool,
    json: bool,
    n_ctx: u32,
    n_threads: Option<i32>,
    n_gpu_layers: i32,
    embedding_model: Option<String>,
    embedding_tokenizer: Option<String>,
    embedding_dimension: usize,
    embedding_max_tokens: usize,
    top_k_retrieval: usize,
    min_score: Option<f32>,
    context_token_budget: usize,
    mock_llm: bool,
    mock_llm_response: Option<String>,
    retrieval: bool,
    interactive: bool,
    llm_verbose_log: bool,
    config: Option<String>,
}

/// Run the chat command.
///
/// This is the main entry point for `chatvcode chat`.
/// It handles:
/// 1. Checking for index/vector store existence
/// 2. Setting up embedding service
/// 3. Setting up LLM service (real or mock)
/// 4. Building ChatOptions
/// 5. Running RAG chat (streaming or non-streaming)
/// 6. Formatting and displaying results
fn run_chat(args: ChatArgs) -> Result<(), ChatVCodeError> {
    let project_path = PathBuf::from(&args.path);

    // --- Check project path ---
    if !project_path.exists() {
        return Err(ChatVCodeError::invalid_input(format!(
            "Project path does not exist: {}",
            project_path.display()
        )));
    }

    // --- Load configuration file (priority: CLI > local > global > default) ---
    let file_config = if let Some(ref config_path) = args.config {
        let path = PathBuf::from(config_path);
        if !path.exists() {
            return Err(ChatVCodeError::invalid_input(format!(
                "Config file not found: {}",
                path.display()
            )));
        }
        eprintln!("📝 Loading configuration from {}", path.display());
        ChatvcodeConfig::load_from(&path).map_err(|e| {
            ChatVCodeError::internal(format!("Failed to load config file: {e}"))
        })?
    } else {
        // Try local and global config paths with priority: local > global > default
        let local_path = chatvcode_llm::local_config_path();
        let global_path = chatvcode_llm::default_config_path();
        
        let local_exists = local_path.exists();
        let global_exists = global_path.exists();
        
        if local_exists && global_exists {
            eprintln!(
                "📝 Loading configuration: local ({}) > global ({})",
                local_path.display(),
                global_path.display()
            );
        } else if local_exists {
            eprintln!("📝 Loading local configuration from {}", local_path.display());
        } else if global_exists {
            eprintln!("📝 Loading global configuration from {}", global_path.display());
        }
        
        ChatvcodeConfig::load_default().unwrap_or_default()
    };

    // --- Apply config overrides (CLI > config > built-in default) ---
    // For fields where the CLI always provides a default (e.g., temperature=0.7),
    // we treat the clap default as "not set by user" and use the config value
    // only when the CLI value matches the clap default. This is a heuristic,
    // but it provides a reasonable experience for the most common settings.
    let resolved_model = args
        .model
        .clone()
        .or_else(|| file_config.model.path.clone());
    let _resolved_system_prompt = args
        .system_prompt
        .clone()
        .or_else(|| file_config.chat.system_prompt.clone());
    let resolved_n_gpu_layers = file_config.resolve_n_gpu_layers(Some(args.n_gpu_layers));
    let resolved_n_ctx = file_config.resolve_n_ctx(Some(args.n_ctx));
    let resolved_n_threads = args
        .n_threads
        .or(file_config.model.n_threads);
    let resolved_verbose_log = args.llm_verbose_log
        || file_config.model.verbose_log.unwrap_or(false);

    // --- Parse chat template ---
    let chat_template = parse_chat_template(&args.template)?;

    // --- Build generation params ---
    let generation_params = GenerationParams::default()
        .with_temperature(args.temperature)
        .with_top_p(args.top_p)
        .with_top_k(args.top_k)
        .with_max_tokens(args.max_tokens);

    // --- Set up LLM service ---
    let mock_response_text = args.mock_llm_response.clone();
    let mut llm: Box<dyn LlmService> = if args.mock_llm {
        let response_text = mock_response_text.unwrap_or_else(|| {
            "This is a mock response. The LLM-generated answer would appear here when using a real model.".to_string()
        });
        Box::new(MockLlmService::new(response_text))
    } else {
        let model_path = match &resolved_model {
            Some(p) => PathBuf::from(p),
            None => {
                eprintln!("🔍 Auto-discovering model from search directories...");
                let discovered = list_models();
                if discovered.len() == 1 {
                    let p = discovered[0].path.clone();
                    eprintln!("✓ Found model: {} ({})", p.display(), discovered[0].source);
                    p
                } else if discovered.is_empty() {
                    match chatvcode_llm::auto_discover_model() {
                        Ok(p) => {
                            eprintln!("✓ Found model: {}", p.display());
                            p
                        }
                        Err(e) => {
                            eprintln!("✗ Could not auto-discover a model.");
                            eprintln!("  {e}");
                            eprintln!();
                            eprintln!("  Please specify a model with --model=<path>.");
                            return Err(ChatVCodeError::internal(format!(
                                "Model auto-discovery failed: {e}"
                            ))
                            .with_severity(ErrorSeverity::Unrecoverable));
                        }
                    }
                } else {
                    eprintln!("Multiple models found:");
                    for (i, m) in discovered.iter().enumerate() {
                        eprintln!("  [{}] {} ({})", i + 1, m.name, m.source);
                    }
                    eprintln!("Please specify a model with --model=<path>.");
                    // Use first discovered model as fallback
                    discovered[0].path.clone()
                }
            }
        };

        eprintln!("⏳ Loading model: {}...", model_path.display());

        // Show memory estimate before loading
        match estimate_memory(&model_path, resolved_n_ctx) {
            Ok(estimate) => {
                eprintln!(
                    "  Memory estimate: {} (model: {}, KV cache: {}, overhead: {})",
                    format_bytes(estimate.total_bytes),
                    format_bytes(estimate.model_bytes),
                    format_bytes(estimate.kv_cache_bytes),
                    format_bytes(estimate.overhead_bytes),
                );
                if !estimate.fits_in_ram {
                    for w in &estimate.warnings {
                        eprintln!("  ⚠ {w}");
                    }
                    eprintln!("  Hint: reduce --n-ctx or use a smaller model to avoid OOM.");
                }
            }
            Err(e) => {
                log::warn!("Could not estimate memory usage: {e}");
            }
        }

        let n_threads = resolved_n_threads.unwrap_or_else(|| num_cpus::get() as i32);
        let config = LlmConfig::new(&model_path)
            .with_n_ctx(resolved_n_ctx)
            .with_n_threads(n_threads)
            .with_n_gpu_layers(resolved_n_gpu_layers)
            .with_verbose_log(resolved_verbose_log);

        match chatvcode_llm::LlamaService::new(&config) {
            Ok(service) => {
                eprintln!("✓ Model loaded successfully.");
                if let Ok(info) = service.model_info() {
                    eprintln!("  Architecture: {}", info.architecture);
                    eprintln!("  Context size:  {}", info.n_ctx_train);
                    eprintln!(
                        "  Parameters:    {}",
                        chatvcode_llm::format_param_count(info.n_params)
                    );
                    eprintln!("  GPU layers:    {}", resolved_n_gpu_layers);
                }
                Box::new(service) as Box<dyn LlmService>
            }
            Err(e) => {
                eprintln!("✗ Failed to load model: {e}");
                eprintln!();
                eprintln!("  Suggestions:");
                eprintln!("    - Ensure the model file is a valid GGUF format");
                eprintln!("    - Try reducing --n-ctx (current: {})", resolved_n_ctx);
                eprintln!("    - Try --n-gpu-layers=0 for CPU-only mode");
                eprintln!("    - Ensure you have enough RAM/VRAM for the model");
                return Err(ChatVCodeError::internal(format!("Failed to load model: {e}"))
                    .with_severity(ErrorSeverity::Unrecoverable));
            }
        }
    };

    // --- Interactive multi-turn mode ---
    if args.interactive {
        return run_interactive_chat(
            &args,
            &mut llm,
            &chat_template,
            &generation_params,
        );
    }

    // --- LLM-only mode: direct inference without RAG ---
    if !args.retrieval {
        return run_chat_llm_only(&args, &*llm, &chat_template, &generation_params);
    }

    // === Full RAG mode (requires vector store + embedding model) ===

    eprintln!("📂 Project: {}", project_path.display());

    // --- Resolve vector store and metadata paths ---
    let chat_options_template = ChatOptions::new(&project_path);
    let vector_store_path = chat_options_template.resolve_vector_store_path();
    let metadata_store_path = chat_options_template.resolve_metadata_store_path();

    // --- Check for existing index ---
    let has_vectors = vector_store_path.exists();
    let has_metadata = metadata_store_path.exists();

    if !has_vectors {
        eprintln!("⚠ No vector store found at {}", vector_store_path.display());
        eprintln!(
            "  Run `chatvcode index {} --embedding-model=<MODEL>` first to create an index.",
            args.path
        );
        eprintln!();
        eprintln!("  Or use --retrieval=false to skip retrieval and query the LLM directly.");
        return Err(ChatVCodeError::invalid_input(format!(
            "Vector store not found at {}. Index first, or use --retrieval=false.",
            vector_store_path.display()
        )));
    }

    if !has_metadata {
        eprintln!("⚠ No metadata store found at {}", metadata_store_path.display());
        eprintln!("  Chunk resolution may be limited. Consider re-indexing.");
    }

    // --- Build chat options (without embedding config, set later) ---
    // When no explicit token budget is set, derive one from n_ctx:
    // Reserve tokens for the system prompt, chat template formatting, and
    // the model's completion (max_tokens).  This prevents context overflow.
    let context_token_budget = if args.context_token_budget == 0 {
        // Leave headroom: system prompt (~200 tokens) + chat formatting (~100)
        // + model completion (max_tokens).  Fall back to 0 only if n_ctx
        // itself is too small.
        let reserved = generation_params.max_tokens as usize + 300;
        (args.n_ctx as usize).saturating_sub(reserved)
    } else {
        args.context_token_budget
    };

    let mut chat_options = ChatOptions::new(&project_path)
        .with_top_k(args.top_k_retrieval)
        .with_chat_template(chat_template)
        .with_generation_params(generation_params)
        .with_context_token_budget(context_token_budget);

    if let Some(ref system_prompt) = args.system_prompt {
        chat_options = chat_options.system_prompt(system_prompt);
    }
    if let Some(min_score) = args.min_score {
        chat_options = chat_options.with_min_score(min_score);
    }

    // --- Set up embedding service ---
    // Two sources of embeddings:
    //   1) ONNX model via --embedding-model (explicit, traditional RAG)
    //   2) GGUF model via --embedding-model (.gguf) or --model (auto-derived embeddings)
    //
    // Priority:
    //   --embedding-model (GGUF or ONNX) > --model (GGUF) > auto-discover (GGUF) > error
    let embedding_service: Box<dyn EmbeddingService> =
        if let Some(model_path) = &args.embedding_model {
            if is_gguf_path(model_path) {
                let n_threads = args.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
                load_gguf_embedding_service(
                    &PathBuf::from(model_path),
                    args.n_gpu_layers,
                    n_threads,
                    args.llm_verbose_log,
                )?
            } else {
                let embedding_config = {
                    let mut config = EmbeddingConfig::new(
                        model_path,
                        args.embedding_dimension,
                        args.embedding_max_tokens,
                    );
                    if let Some(ref tokenizer) = args.embedding_tokenizer {
                        config = config.with_tokenizer_path(tokenizer);
                    }
                    config
                };

                chat_options = chat_options.embedding_config(embedding_config);

                eprintln!("🔍 Loading ONNX embedding model from {}...", model_path);
                let svc = chatvcode_vdb::OnnxEmbeddingService::new(
                    chat_options.embedding_config.as_ref().unwrap().clone(),
                )
                .map_err(|e| {
                    ChatVCodeError::internal(format!(
                        "Failed to initialize embedding service: {e}"
                    ))
                    .with_severity(ErrorSeverity::Unrecoverable)
                })?;
                eprintln!("✓ ONNX embedding model loaded (dim={}).", svc.dimension());
                Box::new(svc) as Box<dyn EmbeddingService>
            }
        } else if !args.mock_llm {
            // GGUF model embeddings (reuses the LLM model)
            let model_path = match &args.model {
                Some(p) => PathBuf::from(p),
                None => match chatvcode_llm::auto_discover_model() {
                    Ok(p) => p,
                    Err(e) => {
                        return Err(ChatVCodeError::internal(format!(
                            "Cannot use GGUF embeddings without a model. \
                             Model auto-discovery failed: {e}. \
                             Please provide --model=<path> for GGUF-based embeddings, \
                             or --embedding-model=<path> for ONNX embeddings."
                        )));
                    }
                },
            };

            let n_threads = args.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
            eprintln!("🔍 Loading GGUF embedding context from {}...", model_path.display());

            // Load the model again for embedding use
            // (separate context from the inference context, but same model weights)
            let gguf_model = match LlamaModel::load(&model_path, args.n_gpu_layers, true, false) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    return Err(ChatVCodeError::internal(format!(
                        "Failed to load GGUF model for embeddings: {e}. \
                     Try providing --embedding-model for ONNX embeddings instead."
                    ))
                    .with_severity(ErrorSeverity::Unrecoverable));
                }
            };

            match LlamaEmbeddingService::new(gguf_model, 512, n_threads) {
                Ok(svc) => {
                    eprintln!("✓ GGUF embedding service ready (dim={}).", svc.dimension());
                    Box::new(LlamaEmbeddingAdapter::new(svc)) as Box<dyn EmbeddingService>
                }
                Err(e) => {
                    return Err(ChatVCodeError::internal(format!(
                        "Failed to create GGUF embedding context: {e}. \
                     Try providing --embedding-model for ONNX embeddings instead, \
                     or use --retrieval=false to skip retrieval."
                    ))
                    .with_severity(ErrorSeverity::Unrecoverable));
                }
            }
        } else {
            return Err(ChatVCodeError::invalid_input(
                "No embedding model available for RAG retrieval. Provide one of:\n\
             --embedding-model=<path>  Use an ONNX model for embeddings\n\
             --model=<path>            Reuse the GGUF LLM model for embeddings\n\
             --retrieval=false          Skip retrieval and query the LLM directly",
            ));
        };

    // --- Run chat ---
    if !args.stream {
        run_chat_sync(&args, &*llm, &*embedding_service, &chat_options)
    } else {
        run_chat_streaming(&args, &*llm, &*embedding_service, &chat_options)
    }
}

/// Parse a chat template string from CLI argument.
fn parse_chat_template(template: &str) -> Result<ChatTemplate, ChatVCodeError> {
    match template.to_lowercase().as_str() {
        "auto" => Ok(ChatTemplate::Auto),
        "raw" => Ok(ChatTemplate::Raw),
        "chatml" => Ok(ChatTemplate::ChatML),
        "llama3" | "llama-3" => Ok(ChatTemplate::Llama3),
        "deepseek" | "deepseek2" | "deepseek3" | "deepseek-v3" => Ok(ChatTemplate::DeepSeek),
        _ => Ok(ChatTemplate::Custom(template.to_string())),
    }
}

/// Index source files using a GGUF model for embedding generation.
///
/// This function auto-discovers the GGUF model if not provided, loads it for
/// embedding, and delegates to [`chatvcode_core::index_with_embedding_service`].
fn index_with_gguf(
    path: &str,
    model: Option<&str>,
    options: &IndexOptions,
    batch_size: usize,
    vector_store_path: Option<&str>,
    n_threads: Option<i32>,
    n_gpu_layers: i32,
    llm_verbose_log: bool,
) -> Result<IndexResult, ChatVCodeError> {
    // Resolve model path
    let model_path = match model {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("🔍 Auto-discovering GGUF model from ~/.chatvcode/models/...");
            match chatvcode_llm::auto_discover_model() {
                Ok(p) => {
                    eprintln!("✓ Found model: {}", p.display());
                    p
                }
                Err(e) => {
                    eprintln!("✗ Could not auto-discover a model.");
                    eprintln!("  {e}");
                    return Err(ChatVCodeError::internal(format!(
                        "Model auto-discovery failed: {e}"
                    ))
                    .with_severity(ErrorSeverity::Unrecoverable));
                }
            }
        }
    };

    let n_threads_val = n_threads.unwrap_or_else(|| num_cpus::get() as i32);

    eprintln!("⏳ Loading GGUF model for embedding: {}...", model_path.display());

    // Use a smaller context window for embedding: most code chunks (functions,
    // structs, etc.) are well under 128 tokens.  A smaller n_ctx dramatically
    // reduces computation time, especially on CPU-only builds without CUDA.
    let embed_svc = chatvcode_llm::LlamaEmbeddingService::from_path(
        &model_path,
        128,
        n_threads_val,
        n_gpu_layers,
        llm_verbose_log,
    )
    .map_err(|e| {
        ChatVCodeError::internal(format!("Failed to load GGUF model for embedding: {e}"))
            .with_severity(ErrorSeverity::Unrecoverable)
    })?;

    eprintln!("✓ GGUF embedding service ready (dim={}).", embed_svc.dimension());

    let adapter = LlamaEmbeddingAdapter::new(embed_svc);

    // Build synthetic EmbeddingOptions for VDB path and batch size.
    // The config fields are not used (embedding comes from the GGUF adapter),
    // but we need a valid EmbeddingOptions for run_embedding_with_service.
    let vstore_path = vector_store_path
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(path).join(".chatvcode").join("vectors.vdb"));

    let mut opts_with_embedding = options.clone();
    opts_with_embedding.embedding = Some(chatvcode_core::EmbeddingOptions {
        config: EmbeddingConfig::new("", 0, 512),
        vector_store_path: vstore_path,
        batch_size,
    });

    chatvcode_core::index_with_embedding_service(
        path,
        &parse_source,
        &opts_with_embedding,
        &adapter,
    )
}

/// Run chat in interactive multi-turn REPL mode.
///
/// Creates a [`ChatSession`] and enters a read-eval-print loop where each
/// user message is answered with conversation history preserved across turns.
///
/// Uses `rustyline` for line editing, history persistence, and tab completion.
///
/// Supported commands:
/// - `/quit`, `/q` — Exit interactive mode
/// - `/help`, `/h` — Show available commands
/// - `/clear` — Clear conversation history
/// - `/sources`, `/src` — Show sources from the last response
/// - `/retry`, `/r` — Resend the last question
/// - `/save [path]` — Save session to JSON file
/// - `/load [path]` — Restore session from JSON file
/// - `/history` — Show conversation history summary
/// - `/model list` — List available models
/// - `/model info [path]` — Show model metadata
/// - `/model switch <path>` — Switch to a different model
/// - `/model memory [path]` — Estimate memory usage
/// - `/model gpu [path]` — Recommend GPU layers
///
/// When `--retrieval=false` (LLM-only mode), queries go directly to the LLM.
/// When retrieval is enabled, each turn performs a fresh semantic search.
fn run_interactive_chat(
    args: &ChatArgs,
    llm: &mut Box<dyn LlmService>,
    chat_template: &ChatTemplate,
    generation_params: &GenerationParams,
) -> Result<(), ChatVCodeError> {
    eprintln!();
    eprintln!("🎤 Interactive chat mode (type `/quit` to exit, `/help` for commands)");
    eprintln!("📂 Project: {}", args.path);
    eprintln!(
        "   Mode: {}",
        if args.retrieval { "RAG (with code context)" } else { "LLM only" }
    );
    eprintln!();

    let mut session = ChatSession::new(chat_template.clone());

    if let Some(ref sys) = args.system_prompt {
        session.set_system_prompt(Some(sys.clone()));
    } else {
        session.set_system_prompt(Some(
            "You are a helpful coding assistant. Answer questions clearly and concisely.".into(),
        ));
    }

    session = session
        .max_context_tokens(args.n_ctx as usize)
        .reserve_for_response(args.max_tokens.max(1024) as usize);

    let embedding_service: Option<Box<dyn EmbeddingService>> = if args.retrieval {
        match setup_embedding_for_interactive(args) {
            Ok(svc) => Some(svc),
            Err(e) => {
                eprintln!("⚠ Embedding service unavailable: {e}");
                eprintln!("  Falling back to LLM-only mode.");
                None
            }
        }
    } else {
        None
    };

    let mut last_sources: Vec<SourceReference> = Vec::new();
    let mut last_question: Option<String> = None;

    let mut rl = DefaultEditor::new().map_err(|e| {
        ChatVCodeError::internal(format!("Failed to initialize line editor: {e}"))
    })?;

    let history_path = interactive_history_path();
    if let Some(ref hp) = history_path {
        if rl.load_history(hp).is_err() {
            // First run — no history file yet
        }
    }

    let default_session_path = interactive_default_session_path();

    loop {
        let readline = rl.readline("💬 > ");

        match readline {
            Ok(line) => {
                let input = line.trim().to_string();
                if input.is_empty() {
                    continue;
                }

                rl.add_history_entry(&input).ok();

                // --- Handle commands ---
                if input.starts_with('/') {
                    let cmd_result = handle_interactive_command(
                        &input,
                        &mut session,
                        &mut last_sources,
                        &mut last_question,
                        chat_template,
                        &default_session_path,
                        args,
                    );
                    match cmd_result {
                        InteractiveAction::Continue => continue,
                        InteractiveAction::Quit => break,
                        InteractiveAction::ProcessQuestion(q) => {
                            // /retry — fall through to inference with q
                            run_interactive_turn(
                                &q,
                                &**llm,
                                &embedding_service,
                                args,
                                generation_params,
                                &mut session,
                                &mut last_sources,
                                &mut last_question,
                            );
                        }
                        InteractiveAction::SwitchModel(new_path) => {
                            match switch_model(&new_path, args) {
                                Ok(new_llm) => {
                                    *llm = new_llm;
                                    eprintln!("✓ Model switched successfully.");
                                    session.clear();
                                    eprintln!("  Conversation history cleared.");
                                }
                                Err(e) => {
                                    eprintln!("✗ Failed to switch model: {e}");
                                }
                            }
                        }
                    }
                } else {
                    last_question = Some(input.clone());
                    run_interactive_turn(
                        &input,
                        &**llm,
                        &embedding_service,
                        args,
                        generation_params,
                        &mut session,
                        &mut last_sources,
                        &mut last_question,
                    );
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                eprintln!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                eprintln!("👋 Goodbye!");
                break;
            }
            Err(e) => {
                eprintln!("✗ Input error: {e}");
                break;
            }
        }
    }

    if let Some(ref hp) = history_path {
        rl.save_history(hp).ok();
    }

    Ok(())
}

/// Result of processing an interactive command.
enum InteractiveAction {
    Continue,
    Quit,
    ProcessQuestion(String),
    SwitchModel(String),
}

/// Resolve the history file path (~/.chatvcode/history).
fn interactive_history_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".chatvcode").join("history"))
}

/// Resolve the default session save path (~/.chatvcode/session.json).
fn interactive_default_session_path() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".chatvcode").join("session.json"))
        .unwrap_or_else(|| PathBuf::from("session.json"))
}

/// Handle an interactive slash command.
fn handle_interactive_command(
    input: &str,
    session: &mut ChatSession,
    last_sources: &mut Vec<SourceReference>,
    last_question: &mut Option<String>,
    chat_template: &ChatTemplate,
    default_session_path: &PathBuf,
    args: &ChatArgs,
) -> InteractiveAction {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).filter(|s| !s.is_empty());

    match cmd {
        "/quit" | "/exit" | "/q" => {
            eprintln!("👋 Goodbye!");
            InteractiveAction::Quit
        }
        "/help" | "/h" | "/?" => {
            print_interactive_help();
            InteractiveAction::Continue
        }
        "/clear" => {
            session.clear();
            last_sources.clear();
            *last_question = None;
            eprintln!("✓ Conversation history cleared.");
            InteractiveAction::Continue
        }
        "/sources" | "/src" => {
            display_interactive_sources(last_sources);
            InteractiveAction::Continue
        }
        "/retry" | "/r" => {
            if let Some(q) = last_question {
                eprintln!("🔄 Retrying: {q}");
                InteractiveAction::ProcessQuestion(q.clone())
            } else {
                eprintln!("(No previous question to retry)");
                InteractiveAction::Continue
            }
        }
        "/save" => {
            let path = arg
                .map(PathBuf::from)
                .unwrap_or_else(|| default_session_path.clone());
            match session.to_json() {
                Ok(json) => match std::fs::write(&path, &json) {
                    Ok(()) => {
                        eprintln!("✓ Session saved to {}", path.display());
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to save session: {e}");
                    }
                },
                Err(e) => {
                    eprintln!("✗ Failed to serialize session: {e}");
                }
            }
            InteractiveAction::Continue
        }
        "/load" => {
            let path = arg
                .map(PathBuf::from)
                .unwrap_or_else(|| default_session_path.clone());
            match std::fs::read_to_string(&path) {
                Ok(json) => match ChatSession::from_json(&json, chat_template.clone()) {
                    Ok(loaded) => {
                        let turns = loaded.turn_count();
                        let tokens = loaded.estimated_tokens();
                        *session = loaded
                            .max_context_tokens(session.context_token_limit())
                            .reserve_for_response(session.response_token_reserve());
                        last_sources.clear();
                        *last_question = None;
                        eprintln!(
                            "✓ Session loaded from {} ({} turns, ~{} tokens)",
                            path.display(),
                            turns,
                            tokens
                        );
                    }
                    Err(e) => {
                        eprintln!("✗ Failed to deserialize session: {e}");
                    }
                },
                Err(e) => {
                    eprintln!("✗ Failed to read session file: {e}");
                }
            }
            InteractiveAction::Continue
        }
        "/history" => {
            display_interactive_history(session);
            InteractiveAction::Continue
        }
        "/model" => {
            let subcmd = arg.unwrap_or("list");
            let subcmd_parts: Vec<&str> = subcmd.splitn(2, ' ').collect();
            let action = subcmd_parts[0];
            let action_arg = subcmd_parts.get(1).map(|s| s.trim()).filter(|s| !s.is_empty());

            match action {
                "list" | "ls" => {
                    let models = list_models();
                    if models.is_empty() {
                        eprintln!("No GGUF models found in search directories.");
                        eprintln!("Search paths:");
                        for (dir, source) in chatvcode_llm::model_search_dirs() {
                            eprintln!("  {} ({})", dir.display(), source);
                        }
                    } else {
                        eprintln!("Discovered {} model(s):\n", models.len());
                        for (i, model) in models.iter().enumerate() {
                            eprintln!("[{}] {}", i + 1, model.summary_line());
                            eprintln!("    Path: {}", model.path.display());
                            eprintln!();
                        }
                        eprintln!("Use `/model switch <path>` to switch models.");
                    }
                    InteractiveAction::Continue
                }
                "info" => {
                    if let Some(path_str) = action_arg {
                        let model_path = PathBuf::from(path_str);
                        if !model_path.exists() {
                            eprintln!("✗ Model file not found: {}", model_path.display());
                        } else {
                            match chatvcode_llm::pre_validate_model(&model_path) {
                                Ok(meta) => {
                                    println!("{}", chatvcode_llm::format_gguf_summary(&model_path, &meta));
                                }
                                Err(e) => {
                                    eprintln!("✗ Failed to read model metadata: {e}");
                                }
                            }
                        }
                    } else {
                        eprintln!("Usage: /model info <path>");
                    }
                    InteractiveAction::Continue
                }
                "switch" => {
                    if let Some(path_str) = action_arg {
                        let model_path = PathBuf::from(path_str);
                        if !model_path.exists() {
                            eprintln!("✗ Model file not found: {}", model_path.display());
                            InteractiveAction::Continue
                        } else {
                            eprintln!("⚙ Switching to model: {}", model_path.display());
                            InteractiveAction::SwitchModel(path_str.to_string())
                        }
                    } else {
                        eprintln!("Usage: /model switch <path>");
                        eprintln!("Use `/model list` to see available models.");
                        InteractiveAction::Continue
                    }
                }
                "memory" => {
                    if let Some(path_str) = action_arg {
                        let model_path = PathBuf::from(path_str);
                        if !model_path.exists() {
                            eprintln!("✗ Model file not found: {}", model_path.display());
                        } else {
                            let n_ctx = args.n_ctx;
                            match estimate_memory(&model_path, n_ctx) {
                                Ok(estimate) => {
                                    eprintln!("Memory Estimation for {}", model_path.display());
                                    eprintln!("Context size: {n_ctx} tokens\n");
                                    eprintln!("{}", estimate.summary());
                                }
                                Err(e) => {
                                    eprintln!("✗ Failed to estimate memory: {e}");
                                }
                            }
                        }
                    } else {
                        eprintln!("Usage: /model memory <path>");
                    }
                    InteractiveAction::Continue
                }
                "gpu" => {
                    if let Some(path_str) = action_arg {
                        let model_path = PathBuf::from(path_str);
                        if !model_path.exists() {
                            eprintln!("✗ Model file not found: {}", model_path.display());
                        } else {
                            match recommend_gpu_layers(&model_path, None) {
                                Ok(rec) => {
                                    eprintln!("GPU Layer Recommendation for {}", model_path.display());
                                    eprintln!("{}", rec.summary());
                                }
                                Err(e) => {
                                    eprintln!("✗ Failed to recommend GPU layers: {e}");
                                }
                            }
                        }
                    } else {
                        eprintln!("Usage: /model gpu <path>");
                    }
                    InteractiveAction::Continue
                }
                _ => {
                    eprintln!("Unknown model command: {action}");
                    eprintln!("Available: list, info, switch, memory, gpu");
                    InteractiveAction::Continue
                }
            }
        }
        _ => {
            eprintln!("Unknown command: {cmd}. Type /help for commands.");
            InteractiveAction::Continue
        }
    }
}

/// Switch the LLM model at runtime in interactive mode.
///
/// Creates a new [`LlamaService`] from the given model path using the
/// current CLI configuration (n_ctx, n_threads, n_gpu_layers, etc.).
fn switch_model(
    model_path: &str,
    args: &ChatArgs,
) -> Result<Box<dyn LlmService>, ChatVCodeError> {
    let path = PathBuf::from(model_path);

    if !path.exists() {
        return Err(ChatVCodeError::invalid_input(format!(
            "Model file not found: {}",
            path.display()
        )));
    }

    // Show memory estimate before loading
    match estimate_memory(&path, args.n_ctx) {
        Ok(estimate) => {
            eprintln!(
                "  Memory estimate: {}",
                format_bytes(estimate.total_bytes)
            );
            if !estimate.fits_in_ram {
                for w in &estimate.warnings {
                    eprintln!("  ⚠ {w}");
                }
            }
        }
        Err(e) => {
            log::warn!("Could not estimate memory usage: {e}");
        }
    }

    let n_threads = args.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
    let config = LlmConfig::new(&path)
        .with_n_ctx(args.n_ctx)
        .with_n_threads(n_threads)
        .with_n_gpu_layers(args.n_gpu_layers)
        .with_verbose_log(args.llm_verbose_log);

    let service = LlamaService::new(&config).map_err(|e| {
        ChatVCodeError::internal(format!("Failed to load model '{}': {e}", path.display()))
            .with_severity(ErrorSeverity::Unrecoverable)
    })?;

    if let Ok(info) = service.model_info() {
        eprintln!("  Architecture: {}", info.architecture);
        eprintln!("  Context size:  {}", info.n_ctx_train);
        eprintln!(
            "  Parameters:    {}",
            chatvcode_llm::format_param_count(info.n_params)
        );
        eprintln!("  GPU layers:    {}", args.n_gpu_layers);
    }

    Ok(Box::new(service) as Box<dyn LlmService>)
}

/// Run a single interactive turn (RAG or LLM-only).
fn run_interactive_turn(
    question: &str,
    llm: &dyn LlmService,
    embedding_service: &Option<Box<dyn EmbeddingService>>,
    args: &ChatArgs,
    generation_params: &GenerationParams,
    session: &mut ChatSession,
    last_sources: &mut Vec<SourceReference>,
    last_question: &mut Option<String>,
) {
    last_sources.clear();
    *last_question = Some(question.to_string());

    if let Some(embed_svc) = embedding_service {
        if args.stream {
            match run_interactive_rag_turn_stream(
                question,
                llm,
                &**embed_svc,
                args,
                generation_params,
                session,
            ) {
                Ok((response, sources)) => {
                    *last_sources = sources;
                    session.add_user_message(question);
                    session.add_assistant_message(&response);
                }
                Err(e) => {
                    eprintln!("✗ RAG chat failed: {e}");
                }
            }
        } else {
            match run_interactive_rag_turn(
                question,
                llm,
                &**embed_svc,
                args,
                generation_params,
                session,
            ) {
                Ok((response, sources)) => {
                    *last_sources = sources;
                    display_interactive_response(&response, last_sources, false);
                    session.add_user_message(question);
                    session.add_assistant_message(&response);
                }
                Err(e) => {
                    eprintln!("✗ RAG chat failed: {e}");
                }
            }
        }
    } else if args.stream {
        match session.chat_stream(question, llm, generation_params) {
            Ok(rx) => {
                eprintln!("--- Response ---");
                let mut full_text = String::new();
                let mut token_count = 0u32;
                let stdout = io::stdout();
                let mut handle = stdout.lock();

                loop {
                    match rx.recv_timeout(std::time::Duration::from_secs(120)) {
                        Ok(event) => match event {
                            StreamEvent::Started => {}
                            StreamEvent::Token(token) => {
                                print!("{token}");
                                handle.flush().ok();
                                full_text.push_str(&token);
                                token_count += 1;
                            }
                            StreamEvent::Completed => break,
                            StreamEvent::Cancelled => {
                                eprintln!("\n⚠ Generation was cancelled.");
                                break;
                            }
                            StreamEvent::Error(msg) => {
                                eprintln!("\n✗ Error: {msg}");
                                break;
                            }
                        },
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            eprintln!("\n✗ Generation timed out.");
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            break;
                        }
                    }
                }
                drop(handle);
                eprintln!();
                eprintln!("--- End ---");

                if !full_text.is_empty() {
                    session.add_assistant_message(&full_text);
                }
                eprintln!("  Tokens: {token_count}");
            }
            Err(e) => {
                eprintln!("✗ LLM error: {e}");
            }
        }
    } else {
        match session.chat(question, llm, generation_params) {
            Ok(response) => {
                display_interactive_response(&response.text, &[], false);
            }
            Err(e) => {
                eprintln!("✗ LLM error: {e}");
            }
        }
    }
}

/// Set up embedding service for interactive chat.
fn setup_embedding_for_interactive(
    args: &ChatArgs,
) -> Result<Box<dyn EmbeddingService>, ChatVCodeError> {
    if args.mock_llm {
        return Err(ChatVCodeError::invalid_input(
            "RAG mode is not supported with --mock-llm in interactive mode.",
        ));
    }

    // Check if vector store exists
    let chat_opts = ChatOptions::new(&args.path);
    let vstore_path = chat_opts.resolve_vector_store_path();
    if !vstore_path.exists() {
        return Err(ChatVCodeError::invalid_input(format!(
            "Vector store not found at {}. Run `chatvcode index` first.",
            vstore_path.display()
        )));
    }

    if let Some(model_path) = &args.embedding_model {
        if is_gguf_path(model_path) {
            let n_threads = args.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
            load_gguf_embedding_service(
                &PathBuf::from(model_path),
                args.n_gpu_layers,
                n_threads,
                args.llm_verbose_log,
            )
        } else {
            let embedding_config = {
                let mut config = EmbeddingConfig::new(
                    model_path,
                    args.embedding_dimension,
                    args.embedding_max_tokens,
                );
                if let Some(ref tokenizer) = args.embedding_tokenizer {
                    config = config.with_tokenizer_path(tokenizer);
                }
                config
            };
            let svc = chatvcode_vdb::OnnxEmbeddingService::new(embedding_config)
                .map_err(|e| ChatVCodeError::internal(format!("ONNX embedding init failed: {e}")))?;
            Ok(Box::new(svc))
        }
    } else if let Some(model_path) = &args.model {
        let n_threads = args.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
        eprintln!("🔍 Loading GGUF embedding model from {}...", model_path);
        let embed_svc = LlamaEmbeddingService::from_path(
            &PathBuf::from(model_path),
            512,
            n_threads,
            args.n_gpu_layers,
            args.llm_verbose_log,
        )
        .map_err(|e| {
            ChatVCodeError::internal(format!("Failed to load GGUF embedding model: {e}"))
        })?;
        eprintln!("✓ GGUF embedding model loaded (dim={}).", embed_svc.dimension());
        Ok(Box::new(LlamaEmbeddingAdapter::new(embed_svc)))
    } else {
        // Auto-discover
        match chatvcode_llm::auto_discover_model() {
            Ok(model_path) => {
                let n_threads = args.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
                eprintln!("🔍 Auto-loading GGUF embedding model from {}...", model_path.display());
                let embed_svc = LlamaEmbeddingService::from_path(
                    &model_path,
                    512,
                    n_threads,
                    args.n_gpu_layers,
                    args.llm_verbose_log,
                )
                .map_err(|e| {
                    ChatVCodeError::internal(format!("Failed to load GGUF embedding model: {e}"))
                })?;
                eprintln!("✓ GGUF embedding model loaded (dim={}).", embed_svc.dimension());
                Ok(Box::new(LlamaEmbeddingAdapter::new(embed_svc)))
            }
            Err(e) => Err(ChatVCodeError::invalid_input(format!(
                "No embedding model available: {e}"
            ))),
        }
    }
}

/// Run a single RAG turn within interactive chat.
///
/// Performs semantic search, builds a RAG prompt with conversation history,
/// runs inference, and returns the answer with sources.
fn run_interactive_rag_turn(
    question: &str,
    llm: &dyn LlmService,
    embedding_service: &dyn EmbeddingService,
    args: &ChatArgs,
    generation_params: &GenerationParams,
    session: &ChatSession,
) -> Result<(String, Vec<SourceReference>), ChatVCodeError> {
    let context_token_budget = if args.context_token_budget == 0 {
        let reserved = generation_params.max_tokens as usize + 300;
        (args.n_ctx as usize).saturating_sub(reserved)
    } else {
        args.context_token_budget
    };

    let mut chat_options = ChatOptions::new(&args.path)
        .with_top_k(args.top_k_retrieval)
        .with_chat_template(session.template().clone())
        .with_generation_params(generation_params.clone())
        .with_context_token_budget(context_token_budget);

    if let Some(ref sys) = args.system_prompt {
        chat_options = chat_options.system_prompt(sys);
    }
    if let Some(min_score) = args.min_score {
        chat_options = chat_options.with_min_score(min_score);
    }

    let response = chat_with_context(question, llm, embedding_service, &chat_options)?;

    Ok((response.answer, response.sources))
}

/// Run a streaming RAG turn within interactive chat.
///
/// Uses `chat_with_context_stream` so tokens are displayed in real-time
/// instead of blocking until the full response is ready.
fn run_interactive_rag_turn_stream(
    question: &str,
    llm: &dyn LlmService,
    embedding_service: &dyn EmbeddingService,
    args: &ChatArgs,
    generation_params: &GenerationParams,
    session: &ChatSession,
) -> Result<(String, Vec<SourceReference>), ChatVCodeError> {
    let context_token_budget = if args.context_token_budget == 0 {
        let reserved = generation_params.max_tokens as usize + 300;
        (args.n_ctx as usize).saturating_sub(reserved)
    } else {
        args.context_token_budget
    };

    let mut chat_options = ChatOptions::new(&args.path)
        .with_top_k(args.top_k_retrieval)
        .with_chat_template(session.template().clone())
        .with_generation_params(generation_params.clone())
        .with_context_token_budget(context_token_budget);

    if let Some(ref sys) = args.system_prompt {
        chat_options = chat_options.system_prompt(sys);
    }
    if let Some(min_score) = args.min_score {
        chat_options = chat_options.with_min_score(min_score);
    }

    let streaming =
        chat_with_context_stream(question, llm, embedding_service, &chat_options)?;

    if !streaming.sources.is_empty() {
        eprintln!();
        eprintln!("📚 Using {} context snippet(s):", streaming.sources.len());
        for (i, src) in streaming.sources.iter().enumerate() {
            eprintln!("  [{}] {} (score: {:.3})", i + 1, src.display_path(), src.score);
        }
        eprintln!();
    }

    eprintln!("--- Response ---");
    let mut full_text = String::new();
    let mut token_count = 0u32;
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    let rx = streaming.event_receiver;
    loop {
        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(event) => match event {
                StreamEvent::Started => {}
                StreamEvent::Token(token) => {
                    print!("{token}");
                    handle.flush().ok();
                    full_text.push_str(&token);
                    token_count += 1;
                }
                StreamEvent::Completed => break,
                StreamEvent::Cancelled => {
                    eprintln!("\n⚠ Generation was cancelled.");
                    break;
                }
                StreamEvent::Error(msg) => {
                    eprintln!("\n✗ Error: {msg}");
                    break;
                }
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("\n✗ Generation timed out (120s).");
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
    drop(handle);
    eprintln!();
    eprintln!("--- End ---");
    eprintln!("  Tokens: ~{token_count}");

    Ok((full_text, streaming.sources))
}

/// Display a response in interactive mode.
fn display_interactive_response(answer: &str, sources: &[SourceReference], _json: bool) {
    eprintln!("--- Response ---");
    println!("{answer}");
    eprintln!("--- End ---");
    display_interactive_sources(sources);
    eprintln!();
}

/// Display source references in interactive mode.
fn display_interactive_sources(sources: &[SourceReference]) {
    if sources.is_empty() {
        eprintln!("  (No sources from the last response)");
    } else {
        eprintln!("📎 Sources ({n}):", n = sources.len());
        for (i, src) in sources.iter().enumerate() {
            eprintln!(
                "  [{idx}] {path}:{line} ({kind}) [score: {score:.3}]",
                idx = i + 1,
                path = src.file_path.display(),
                line = src.start_line,
                kind = src.symbol_name.as_deref().unwrap_or("unknown"),
                score = src.score,
            );
        }
    }
}

/// Display conversation history summary in interactive mode.
fn display_interactive_history(session: &ChatSession) {
    let turns = session.turn_count();
    let tokens = session.estimated_tokens();
    let messages = session.messages();

    if messages.is_empty() {
        eprintln!("(No conversation history)");
        return;
    }

    eprintln!("📜 History: {} turns, ~{} tokens", turns, tokens);
    for (i, msg) in messages.iter().enumerate() {
        let role = if msg.role == "user" { "You" } else { "Assistant" };
        let preview: String = msg.content.chars().take(80).collect();
        let suffix = if msg.content.len() > 80 { "..." } else { "" };
        eprintln!("  [{}] {}: {}{}", i + 1, role, preview, suffix);
    }
}

/// Print help for interactive chat commands.
fn print_interactive_help() {
    eprintln!();
    eprintln!("  Interactive Chat Commands:");
    eprintln!("    /quit, /q           Exit interactive mode");
    eprintln!("    /help, /h           Show this help");
    eprintln!("    /clear              Clear conversation history");
    eprintln!("    /sources, /src      Show sources from last response");
    eprintln!("    /retry, /r          Resend the last question");
    eprintln!("    /save [path]        Save session to JSON (default: ~/.chatvcode/session.json)");
    eprintln!("    /load [path]        Load session from JSON (default: ~/.chatvcode/session.json)");
    eprintln!("    /history            Show conversation history summary");
    eprintln!();
    eprintln!("  Model Management Commands:");
    eprintln!("    /model list         List available models");
    eprintln!("    /model info <path>  Show model metadata");
    eprintln!("    /model switch <path> Switch to a different model");
    eprintln!("    /model memory <path> Estimate memory usage");
    eprintln!("    /model gpu <path>   Recommend GPU layers");
    eprintln!();
    eprintln!("  Or just type your question to ask the model.");
    eprintln!();
}

/// Run chat in LLM-only mode: direct LLM inference without RAG context.
///
/// This mode is activated via --retrieval=false and does not require a vector store
/// or embedding model. It sends the user question directly to the LLM with an optional
/// system prompt.
fn run_chat_llm_only(
    args: &ChatArgs,
    llm: &dyn LlmService,
    chat_template: &ChatTemplate,
    generation_params: &GenerationParams,
) -> Result<(), ChatVCodeError> {
    eprintln!();
    eprintln!("💬 Question: {}", args.question);
    eprintln!("   Mode: LLM only (no RAG context)");
    eprintln!();

    // Build prompt without RAG context
    let mut builder = ChatPromptBuilder::new(chat_template.clone())
        .user_question(&args.question)
        .add_generation_prompt(true);

    if let Some(ref system_prompt) = args.system_prompt {
        builder = builder.system_prompt(system_prompt);
    } else {
        builder = builder.system_prompt(
            "You are a helpful coding assistant. Answer questions clearly and concisely.",
        );
    }

    let prompt = builder.build().map_err(|e| {
        ChatVCodeError::internal(format!("Failed to build prompt: {e}"))
            .with_severity(ErrorSeverity::Unrecoverable)
    })?;

    if !args.stream {
        // --- Non-streaming mode ---
        let start = Instant::now();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let response = llm
            .infer(&prompt, generation_params, Some(&cancel))
            .map_err(|e| {
                ChatVCodeError::internal(format!("LLM inference failed: {e}"))
                    .with_severity(ErrorSeverity::Unrecoverable)
            })?;
        let total_elapsed = start.elapsed();

        if args.json {
            // Build a minimal ChatResponse-like JSON for consistency
            let json_response = serde_json::json!({
                "answer": response.text,
                "sources": [],
                "token_usage": {
                    "prompt_tokens": response.token_usage.prompt_tokens,
                    "completion_tokens": response.token_usage.completion_tokens,
                    "total_tokens": response.token_usage.total_tokens,
                },
                "stop_reason": format!("{:?}", response.stop_reason),
                "duration_ms": total_elapsed.as_millis(),
                "retrieved_count": 0,
                "used_count": 0,
                "no_context": true,
            });
            println!("{}", serde_json::to_string_pretty(&json_response).unwrap());
        } else {
            println!("{}", response.text);
            println!();
            println!("⏱ Time: {:.1}s", total_elapsed.as_secs_f64());
            println!(
                "📊 Tokens: {} prompt + {} completion = {} total",
                response.token_usage.prompt_tokens,
                response.token_usage.completion_tokens,
                response.token_usage.total_tokens,
            );
            println!("🛑 Stop reason: {:?}", response.stop_reason);
            println!("📚 Mode: LLM only (no RAG context)");
        }
    } else {
        // --- Streaming mode ---
        eprintln!("--- Response ---");
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let start = Instant::now();

        let rx = llm
            .infer_stream(&prompt, generation_params, Some(cancel))
            .map_err(|e| {
                ChatVCodeError::internal(format!("LLM streaming failed: {e}"))
                    .with_severity(ErrorSeverity::Unrecoverable)
            })?;

        let mut full_text = String::new();
        let mut token_count = 0u32;
        let stdout = io::stdout();
        let mut handle = stdout.lock();

        loop {
            match rx.recv_timeout(std::time::Duration::from_secs(120)) {
                Ok(event) => match event {
                    StreamEvent::Started => {}
                    StreamEvent::Token(token) => {
                        print!("{token}");
                        handle.flush().ok();
                        full_text.push_str(&token);
                        token_count += 1;
                    }
                    StreamEvent::Completed => break,
                    StreamEvent::Cancelled => {
                        eprintln!("\n⚠ Generation was cancelled.");
                        break;
                    }
                    StreamEvent::Error(msg) => {
                        eprintln!("\n✗ Error during generation: {msg}");
                        return Err(ChatVCodeError::internal(format!("Generation error: {msg}")));
                    }
                },
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!("\n✗ Generation timed out (120s).");
                    return Err(ChatVCodeError::internal("Generation timed out"));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("\n⚠ Generation channel disconnected.");
                    break;
                }
            }
        }
        drop(handle);

        eprintln!();
        eprintln!("--- End ---");

        let total_elapsed = start.elapsed();

        if args.json {
            let json_response = serde_json::json!({
                "answer": full_text,
                "sources": [],
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": token_count,
                    "total_tokens": token_count,
                },
                "stop_reason": "Eos",
                "duration_ms": total_elapsed.as_millis(),
                "retrieved_count": 0,
                "used_count": 0,
                "no_context": true,
            });
            println!("{}", serde_json::to_string_pretty(&json_response).unwrap());
        } else {
            eprintln!("⏱ Time: {:.1}s", total_elapsed.as_secs_f64());
            eprintln!("📊 Tokens: ~{token_count}");
            eprintln!("📚 Mode: LLM only (no RAG context)");
        }
    }

    Ok(())
}
fn run_chat_sync(
    args: &ChatArgs,
    llm: &dyn LlmService,
    embedding_service: &dyn chatvcode_vdb::EmbeddingService,
    chat_options: &ChatOptions,
) -> Result<(), ChatVCodeError> {
    let start = Instant::now();
    eprintln!();
    eprintln!("💬 Question: {}", args.question);
    eprintln!();

    let result =
        chat_with_context(&args.question, llm, embedding_service, chat_options).map_err(|e| {
            ChatVCodeError::internal(format!("Chat failed: {e}"))
                .with_severity(ErrorSeverity::Unrecoverable)
        })?;
    let total_elapsed = start.elapsed();

    if args.json {
        print_chat_response_json(&result, total_elapsed);
    } else {
        print_chat_response_human(&result, total_elapsed);
    }

    Ok(())
}

/// Run streaming chat (prints tokens as they arrive).
fn run_chat_streaming(
    args: &ChatArgs,
    llm: &dyn LlmService,
    embedding_service: &dyn chatvcode_vdb::EmbeddingService,
    chat_options: &ChatOptions,
) -> Result<(), ChatVCodeError> {
    let start = Instant::now();
    eprintln!();
    eprintln!("💬 Question: {}", args.question);
    eprintln!();
    eprintln!("--- Response ---");

    let streaming = chat_with_context_stream(&args.question, llm, embedding_service, chat_options)
        .map_err(|e| {
            ChatVCodeError::internal(format!("Chat stream failed: {e}"))
                .with_severity(ErrorSeverity::Unrecoverable)
        })?;

    // Display sources immediately (available before streaming starts)
    if !streaming.sources.is_empty() {
        eprintln!();
        eprintln!("📚 Using {} context snippet(s):", streaming.sources.len());
        for (i, src) in streaming.sources.iter().enumerate() {
            eprintln!("  [{}] {} (score: {:.3})", i + 1, src.display_path(), src.score);
        }
        eprintln!();
    }

    // Stream tokens to stdout
    let mut full_text = String::new();
    let mut token_count = 0u32;

    let rx = streaming.event_receiver;
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    loop {
        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(event) => match event {
                StreamEvent::Started => {
                    // Ignore start event
                }
                StreamEvent::Token(token) => {
                    print!("{token}");
                    handle.flush().ok();
                    full_text.push_str(&token);
                    token_count += 1;
                }
                StreamEvent::Completed => {
                    break;
                }
                StreamEvent::Cancelled => {
                    eprintln!("\n⚠ Generation was cancelled.");
                    break;
                }
                StreamEvent::Error(msg) => {
                    eprintln!("\n✗ Error during generation: {msg}");
                    return Err(ChatVCodeError::internal(format!("Generation error: {msg}")));
                }
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("\n✗ Generation timed out (120s).");
                return Err(ChatVCodeError::internal("Generation timed out"));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("\n⚠ Generation channel disconnected unexpectedly.");
                break;
            }
        }
    }
    drop(handle);

    eprintln!();
    eprintln!("--- End ---");

    let total_elapsed = start.elapsed();

    if args.json {
        // For JSON output with streaming, we need to reconstruct the response
        let response = ChatResponse {
            answer: full_text,
            sources: streaming.sources,
            token_usage: chatvcode_llm::TokenUsage::new(0, token_count as i32),
            stop_reason: chatvcode_llm::StopReason::Eos,
            duration: total_elapsed,
            search_duration: streaming.search_duration,
            inference_duration: total_elapsed - streaming.search_duration,
            retrieved_count: streaming.retrieved_count,
            used_count: streaming.used_count,
        };
        print_chat_response_json(&response, total_elapsed);
    } else {
        // Print sources and stats after streaming
        eprintln!();
        eprintln!("{}", format_sources_display(&streaming.sources));
        eprintln!(
            "⏱ Time: {:.1}s (search: {:.1}s, inference: {:.1}s)",
            total_elapsed.as_secs_f64(),
            streaming.search_duration.as_secs_f64(),
            (total_elapsed - streaming.search_duration).as_secs_f64(),
        );
        eprintln!("📊 Tokens: ~{token_count}");
    }

    Ok(())
}

/// Print a chat response in human-readable format.
fn print_chat_response_human(result: &ChatResponse, total_elapsed: std::time::Duration) {
    // Answer
    println!("{}", result.answer);
    println!();

    // Sources
    println!("{}", result.format_sources());
    println!();

    // Stats
    println!(
        "⏱ Time: {:.1}s (search: {:.1}s, inference: {:.1}s)",
        total_elapsed.as_secs_f64(),
        result.search_duration.as_secs_f64(),
        result.inference_duration.as_secs_f64(),
    );
    println!(
        "📊 Tokens: {} prompt + {} completion = {} total",
        result.token_usage.prompt_tokens,
        result.token_usage.completion_tokens,
        result.token_usage.total_tokens,
    );
    println!("🛑 Stop reason: {:?}", result.stop_reason);
    println!(
        "📚 Context: {}/{} snippets used ({} retrieved)",
        result.used_count, result.retrieved_count, result.retrieved_count
    );

    if result.is_no_context() {
        println!();
        println!("⚠ No relevant code context was found for this question.");
        println!("  The answer is based on the model's general knowledge only.");
        println!("  Try re-indexing the project or providing a more specific question.");
    }
}

/// Print a chat response in JSON format.
fn print_chat_response_json(result: &ChatResponse, total_elapsed: std::time::Duration) {
    #[derive(serde::Serialize)]
    struct JsonSource {
        chunk_id: String,
        file_path: String,
        kind: String,
        symbol_name: Option<String>,
        start_line: usize,
        end_line: usize,
        score: f32,
        display_path: String,
    }

    #[derive(serde::Serialize)]
    struct JsonResponse {
        answer: String,
        sources: Vec<JsonSource>,
        token_usage: TokenUsageInfo,
        stop_reason: String,
        duration_ms: u64,
        search_duration_ms: u64,
        inference_duration_ms: u64,
        retrieved_count: usize,
        used_count: usize,
        no_context: bool,
    }

    #[derive(serde::Serialize)]
    struct TokenUsageInfo {
        prompt_tokens: i32,
        completion_tokens: i32,
        total_tokens: i32,
    }

    let sources: Vec<JsonSource> = result
        .sources
        .iter()
        .map(|s| JsonSource {
            chunk_id: s.chunk_id.clone(),
            file_path: s.file_path.display().to_string(),
            kind: s.kind.to_string(),
            symbol_name: s.symbol_name.clone(),
            start_line: s.start_line,
            end_line: s.end_line,
            score: s.score,
            display_path: s.display_path(),
        })
        .collect();

    let response = JsonResponse {
        answer: result.answer.clone(),
        sources,
        token_usage: TokenUsageInfo {
            prompt_tokens: result.token_usage.prompt_tokens,
            completion_tokens: result.token_usage.completion_tokens,
            total_tokens: result.token_usage.total_tokens,
        },
        stop_reason: format!("{:?}", result.stop_reason),
        duration_ms: total_elapsed.as_millis() as u64,
        search_duration_ms: result.search_duration.as_millis() as u64,
        inference_duration_ms: result.inference_duration.as_millis() as u64,
        retrieved_count: result.retrieved_count,
        used_count: result.used_count,
        no_context: result.is_no_context(),
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&response)
            .unwrap_or_else(|e| { format!("{{\"error\": \"Failed to serialize JSON: {e}\"}}") })
    );
}

/// Format source references for display in the terminal.
fn format_sources_display(sources: &[SourceReference]) -> String {
    if sources.is_empty() {
        return "📚 No sources available (answer based on model knowledge only)".to_string();
    }

    let mut out = String::new();
    out.push_str("📚 Sources:\n");
    for (i, src) in sources.iter().enumerate() {
        out.push_str(&format!("  [{}] {} (score: {:.3})\n", i + 1, src.display_path(), src.score));
    }

    // Diagnostic: warn if all retrieval scores are very similar, which indicates
    // the embedding model has poor discrimination (common with causal LMs)
    if sources.len() >= 2 {
        let scores: Vec<f32> = sources.iter().map(|s| s.score).collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let score_range = max_score - min_score;
        if score_range < 0.05 {
            out.push_str(
                "\n⚠️  Warning: Retrieval scores are nearly identical (range={score_range:.4}).\n",
            );
            out.push_str("   This indicates the embedding model has poor discrimination.\n");
            out.push_str(
                "   Consider using a dedicated embedding model (e.g., bge-m3, nomic-embed-text)\n",
            );
            out.push_str("   for better retrieval quality.\n");
        }
    }

    out
}

/// Formats an [`IndexResult`] into a human-readable string.
///
/// Includes statistics, language/kind breakdowns, and error summaries.
#[must_use]
pub fn format_index_result(result: &IndexResult) -> String {
    let mut output = String::new();
    let stats = &result.stats;

    output.push_str("Indexing complete.\n\n");
    output.push_str(&format!("  Files scanned : {}\n", stats.total_files));
    output.push_str(&format!("  Files parsed  : {}\n", stats.parsed_files));
    output.push_str(&format!("  Files skipped : {}\n", stats.skipped_files));
    output.push_str(&format!("  Total chunks  : {}\n", stats.total_chunks));
    output.push_str(&format!("  Source bytes  : {}\n", stats.total_source_bytes));
    output.push_str(&format!("  Errors        : {}\n", stats.total_errors));
    output.push_str(&format!("  Elapsed       : {}ms\n", stats.elapsed_ms));

    if stats.embedded_chunks > 0 || stats.embedding_errors > 0 {
        output.push_str(&format!("  Embedded      : {}\n", stats.embedded_chunks));
        output.push_str(&format!("  Emb. errors   : {}\n", stats.embedding_errors));
        output.push_str(&format!("  Emb. dimension: {}\n", stats.embedding_dimension));
    }

    if !stats.files_by_language.is_empty() {
        output.push_str("\nFiles by language:\n");
        let mut lang_stats: Vec<_> = stats.files_by_language.iter().collect();
        lang_stats.sort_by(|a, b| b.1.cmp(a.1));
        for (lang, count) in lang_stats {
            output.push_str(&format!("  {:<12} : {}\n", lang.as_str(), count));
        }
    }

    if !stats.chunks_by_kind.is_empty() {
        output.push_str("\nChunks by kind:\n");
        let mut kind_stats: Vec<_> = stats.chunks_by_kind.iter().collect();
        kind_stats.sort_by(|a, b| b.1.cmp(a.1));
        for (kind, count) in kind_stats {
            output.push_str(&format!("  {:<12} : {}\n", kind.to_string(), count));
        }
    }

    if !result.errors.is_empty() {
        let unrecoverable: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.severity == ErrorSeverity::Unrecoverable)
            .collect();
        let recoverable: Vec<_> = result
            .errors
            .iter()
            .filter(|e| e.severity == ErrorSeverity::Recoverable)
            .collect();

        if !unrecoverable.is_empty() {
            output.push_str("\nFatal errors:\n");
            for err in unrecoverable {
                let path_str = err
                    .context
                    .path
                    .as_ref()
                    .map_or_else(|| "<unknown>".to_string(), |p| p.display().to_string());
                output.push_str(&format!("  {path_str}: {err}\n"));
            }
        }

        if !recoverable.is_empty() {
            output.push_str("\nRecoverable errors:\n");
            for err in recoverable {
                let path_str = err
                    .context
                    .path
                    .as_ref()
                    .map_or_else(|| "<unknown>".to_string(), |p| p.display().to_string());
                output.push_str(&format!("  {path_str}: {err}\n"));
            }
        }
    }

    let files_with_errors: Vec<_> = result
        .files
        .iter()
        .filter(|f| !f.errors.is_empty())
        .collect();

    if !files_with_errors.is_empty() {
        output.push_str("\nParse warnings:\n");
        for file_result in files_with_errors {
            for err in &file_result.errors {
                output.push_str(&format!("  {}: {err}\n", file_result.file.path.display()));
            }
        }
    }

    output
}

/// Prints the formatted [`IndexResult`] to stdout.
pub fn print_index_result(result: &IndexResult) {
    print!("{}", format_index_result(result));
}

/// Formats search results into a human-readable string.
///
/// Displays each result with its similarity score, file path, line range,
/// chunk kind, symbol name, and a code snippet (truncated if too long).
#[must_use]
pub fn format_search_results(query: &str, results: &[chatvcode_core::SearchResult]) -> String {
    let mut output = String::new();

    output.push_str(&format!("Search results for: {query:?}\n\n"));

    if results.is_empty() {
        output.push_str("  No results found.\n");
        return output;
    }

    output.push_str(&format!("  Found {} result(s):\n\n", results.len()));

    for (i, result) in results.iter().enumerate() {
        let chunk = &result.chunk;
        output.push_str(&format!("  --- Result #{} (score: {:.4}) ---\n", i + 1, result.score));
        output.push_str(&format!("  File   : {}\n", chunk.file_path.display()));
        output.push_str(&format!(
            "  Lines  : {}-{}\n",
            chunk.span.start_line + 1,
            chunk.span.end_line + 1
        ));
        output.push_str(&format!("  Kind   : {}\n", chunk.kind));
        if let Some(ref name) = chunk.symbol_name {
            output.push_str(&format!("  Symbol : {name}\n"));
        }

        let snippet = if chunk.source_text.len() > 500 {
            let truncated: String = chunk.source_text.chars().take(500).collect();
            format!("{truncated}...")
        } else {
            chunk.source_text.clone()
        };
        output.push_str("  Code   :\n");
        for line in snippet.lines() {
            output.push_str(&format!("    {line}\n"));
        }
        output.push('\n');
    }

    output
}

/// Prints the formatted search results to stdout.
pub fn print_search_results(query: &str, results: &[chatvcode_core::SearchResult]) {
    print!("{}", format_search_results(query, results));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chatvcode_llm::StopReason;
    use clap::Parser;

    #[test]
    fn test_cli_chat_command_parses_question() {
        let cli = Cli::try_parse_from(["chatvcode", "chat", "What does main do?"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { question, .. } => assert_eq!(question, "What does main do?"),
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_with_path() {
        let cli = Cli::try_parse_from([
            "chatvcode",
            "chat",
            "Explain this code",
            "--path",
            "/my/project",
        ]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { question, path, .. } => {
                assert_eq!(question, "Explain this code");
                assert_eq!(path, "/my/project");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_with_model() {
        let cli =
            Cli::try_parse_from(["chatvcode", "chat", "test", "--model", "/models/codellama.gguf"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { model, .. } => {
                assert_eq!(model, Some("/models/codellama.gguf".to_string()));
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_with_all_options() {
        let cli = Cli::try_parse_from([
            "chatvcode",
            "chat",
            "How is error handling done?",
            "--path",
            "./my-project",
            "--model",
            "/models/qwen.gguf",
            "--temperature",
            "0.5",
            "--max-tokens",
            "1024",
            "--top-k",
            "50",
            "--top-p",
            "0.95",
            "--template",
            "chatml",
            "--system-prompt",
            "You are a Rust expert.",
            "--stream=false",
            "--json",
            "--n-ctx",
            "4096",
            "--n-gpu-layers=-1",
            "--embedding-model",
            "/models/embed.onnx",
            "--top-k-retrieval",
            "5",
            "--min-score",
            "0.7",
        ]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat {
                question,
                path,
                model,
                temperature,
                max_tokens,
                top_k,
                top_p,
                template,
                system_prompt,
                stream,
                json,
                n_ctx,
                n_gpu_layers,
                top_k_retrieval,
                min_score,
                ..
            } => {
                assert_eq!(question, "How is error handling done?");
                assert_eq!(path, "./my-project");
                assert_eq!(model, Some("/models/qwen.gguf".to_string()));
                assert!((temperature - 0.5).abs() < f32::EPSILON);
                assert_eq!(max_tokens, 1024);
                assert_eq!(top_k, 50);
                assert!((top_p - 0.95).abs() < f32::EPSILON);
                assert_eq!(template, "chatml");
                assert_eq!(system_prompt, Some("You are a Rust expert.".to_string()));
                assert!(!stream);
                assert!(json);
                assert_eq!(n_ctx, 4096);
                assert_eq!(n_gpu_layers, -1);
                assert_eq!(top_k_retrieval, 5);
                assert_eq!(min_score, Some(0.7));
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_defaults() {
        let cli = Cli::try_parse_from(["chatvcode", "chat", "test question"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat {
                path,
                temperature,
                max_tokens,
                top_k,
                top_p,
                template,
                stream,
                json,
                n_ctx,
                n_gpu_layers,
                top_k_retrieval,
                context_token_budget,
                mock_llm,
                ..
            } => {
                assert_eq!(path, ".");
                assert!((temperature - 0.7).abs() < f32::EPSILON);
                assert_eq!(max_tokens, 2048);
                assert_eq!(top_k, 40);
                assert!((top_p - 0.9).abs() < f32::EPSILON);
                assert_eq!(template, "auto");
                assert!(stream); // default is true
                assert!(!json);
                assert_eq!(n_ctx, 8192);
                assert_eq!(n_gpu_layers, 0);
                assert_eq!(top_k_retrieval, 16);
                assert_eq!(context_token_budget, 0);
                assert!(!mock_llm);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_retrieval_default_true() {
        // Default: retrieval should be true
        let cli = Cli::try_parse_from(["chatvcode", "chat", "test question"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { retrieval, .. } => {
                assert!(retrieval);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_retrieval_disabled() {
        // --retrieval=false should set retrieval to false
        let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--retrieval=false"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { retrieval, .. } => {
                assert!(!retrieval);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_retrieval_bare_flag() {
        // Bare --retrieval should default to true
        let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--retrieval"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { retrieval, .. } => {
                assert!(retrieval);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_cli_chat_command_stream_bare_flag() {
        // Bare --stream should default to true
        let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--stream"]);
        assert!(cli.is_ok());
        match cli.unwrap().command {
            Commands::Chat { stream, .. } => {
                assert!(stream);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_parse_chat_template_auto() {
        assert_eq!(parse_chat_template("auto").unwrap(), ChatTemplate::Auto);
        assert_eq!(parse_chat_template("Auto").unwrap(), ChatTemplate::Auto);
        assert_eq!(parse_chat_template("AUTO").unwrap(), ChatTemplate::Auto);
    }

    #[test]
    fn test_parse_chat_template_raw() {
        assert_eq!(parse_chat_template("raw").unwrap(), ChatTemplate::Raw);
        assert_eq!(parse_chat_template("Raw").unwrap(), ChatTemplate::Raw);
    }

    #[test]
    fn test_parse_chat_template_chatml() {
        assert_eq!(parse_chat_template("chatml").unwrap(), ChatTemplate::ChatML);
        assert_eq!(parse_chat_template("ChatML").unwrap(), ChatTemplate::ChatML);
    }

    #[test]
    fn test_parse_chat_template_llama3() {
        assert_eq!(parse_chat_template("llama3").unwrap(), ChatTemplate::Llama3);
        assert_eq!(parse_chat_template("llama-3").unwrap(), ChatTemplate::Llama3);
        assert_eq!(parse_chat_template("Llama3").unwrap(), ChatTemplate::Llama3);
    }

    #[test]
    fn test_parse_chat_template_deepseek() {
        assert_eq!(
            parse_chat_template("deepseek").unwrap(),
            ChatTemplate::DeepSeek
        );
        assert_eq!(
            parse_chat_template("deepseek3").unwrap(),
            ChatTemplate::DeepSeek
        );
        assert_eq!(
            parse_chat_template("DeepSeek-V3").unwrap(),
            ChatTemplate::DeepSeek
        );
    }

    #[test]
    fn test_parse_chat_template_custom() {
        let result =
            parse_chat_template("{% for msg in messages %}{{ msg.content }}{% endfor %}").unwrap();
        // Custom template strings are passed through
        assert!(matches!(result, ChatTemplate::Custom(_)));
    }

    #[test]
    fn test_format_sources_display_with_data() {
        let sources = vec![
            SourceReference {
                chunk_id: "id1".to_string(),
                file_path: PathBuf::from("src/main.rs"),
                kind: chatvcode_core::ChunkKind::Function,
                symbol_name: Some("main".to_string()),
                start_line: 10,
                end_line: 20,
                score: 0.95,
                snippet: "fn main() {}".to_string(),
            },
            SourceReference {
                chunk_id: "id2".to_string(),
                file_path: PathBuf::from("src/lib.rs"),
                kind: chatvcode_core::ChunkKind::Struct,
                symbol_name: Some("Config".to_string()),
                start_line: 5,
                end_line: 15,
                score: 0.82,
                snippet: "struct Config {}".to_string(),
            },
        ];

        let output = format_sources_display(&sources);
        assert!(output.contains("📚 Sources:"));
        assert!(output.contains("src/main.rs"));
        assert!(output.contains("0.950"));
        assert!(output.contains("src/lib.rs"));
        assert!(output.contains("0.820"));
    }

    #[test]
    fn test_format_sources_display_empty() {
        let output = format_sources_display(&[]);
        assert!(output.contains("No sources available"));
    }

    // --- Integration-style tests using MockLlmService ---

    #[test]
    fn test_chat_with_mock_llm_builds_prompt() {
        // Test that the ChatPromptBuilder works with RAG context
        let options = ChatOptions::new("/tmp/project")
            .with_chat_template(ChatTemplate::ChatML)
            .system_prompt("You are a helpful coding assistant.");

        let snippets = vec![
            "--- src/main.rs:10-20 (function: hello) [score: 0.900] ---\nfn hello() {}\n---"
                .to_string(),
        ];

        let prompt =
            chatvcode_core::build_rag_prompt("What does hello do?", &snippets, &options).unwrap();
        assert!(prompt.contains("What does hello do?"));
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("<|im_start|>system"));
    }

    #[test]
    fn test_chat_with_mock_llm_no_context() {
        let options = ChatOptions::new("/tmp/project")
            .with_chat_template(ChatTemplate::ChatML)
            .system_prompt("You are a helpful assistant.");

        // Without context, should still produce a valid prompt
        let prompt = chatvcode_core::build_rag_prompt("What is Rust?", &[], &options).unwrap();
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("<|im_start|>user"));
    }

    #[test]
    fn test_mock_llm_service_basic_inference() {
        let service = MockLlmService::new("Rust is a systems programming language.");
        let params = GenerationParams::default();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let response = service
            .infer("What is Rust?", &params, Some(&cancel))
            .unwrap();
        assert_eq!(response.text, "Rust is a systems programming language.");
        assert_eq!(response.stop_reason, StopReason::Eos);
    }

    #[test]
    fn test_chat_response_format_sources() {
        let response = ChatResponse {
            answer: "It does X".to_string(),
            sources: vec![SourceReference {
                chunk_id: "id1".to_string(),
                file_path: PathBuf::from("src/main.rs"),
                kind: chatvcode_core::ChunkKind::Function,
                symbol_name: Some("main".to_string()),
                start_line: 10,
                end_line: 20,
                score: 0.95,
                snippet: "fn main() {}".to_string(),
            }],
            token_usage: chatvcode_llm::TokenUsage::new(50, 20),
            stop_reason: StopReason::Eos,
            duration: std::time::Duration::from_millis(200),
            search_duration: std::time::Duration::from_millis(20),
            inference_duration: std::time::Duration::from_millis(180),
            retrieved_count: 1,
            used_count: 1,
        };
        let formatted = response.format_sources();
        assert!(formatted.contains("Sources:"));
        assert!(formatted.contains("src/main.rs"));
    }

    fn create_test_chat_args() -> ChatArgs {
        ChatArgs {
            question: String::new(),
            path: ".".to_string(),
            model: None,
            temperature: 0.7,
            max_tokens: 2048,
            top_k: 40,
            top_p: 0.9,
            template: "auto".to_string(),
            system_prompt: None,
            stream: false,
            json: false,
            n_ctx: 8192,
            n_threads: None,
            n_gpu_layers: 0,
            embedding_model: None,
            embedding_tokenizer: None,
            embedding_dimension: 0,
            embedding_max_tokens: 512,
            top_k_retrieval: 5,
            min_score: None,
            context_token_budget: 4096,
            mock_llm: false,
            mock_llm_response: None,
            retrieval: false,
            interactive: true,
            llm_verbose_log: false,
            config: None,
        }
    }

    #[test]
    fn test_interactive_command_quit() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        let mut sources = Vec::new();
        let mut last_q = None;
        let default_path = PathBuf::from("/tmp/test_session.json");
        let args = create_test_chat_args();
        let result = handle_interactive_command(
            "/quit",
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &default_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Quit));
    }

    #[test]
    fn test_interactive_command_clear() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_user_message("hello");
        session.add_assistant_message("hi");
        assert_eq!(session.len(), 2);

        let mut sources = Vec::new();
        let mut last_q = Some("hello".to_string());
        let default_path = PathBuf::from("/tmp/test_session.json");
        let args = create_test_chat_args();
        let result = handle_interactive_command(
            "/clear",
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &default_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Continue));
        assert_eq!(session.len(), 0);
        assert!(last_q.is_none());
    }

    #[test]
    fn test_interactive_command_retry_no_history() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        let mut sources = Vec::new();
        let mut last_q: Option<String> = None;
        let default_path = PathBuf::from("/tmp/test_session.json");
        let args = create_test_chat_args();
        let result = handle_interactive_command(
            "/retry",
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &default_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Continue));
    }

    #[test]
    fn test_interactive_command_retry_with_history() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        let mut sources = Vec::new();
        let mut last_q = Some("What is Rust?".to_string());
        let default_path = PathBuf::from("/tmp/test_session.json");
        let args = create_test_chat_args();
        let result = handle_interactive_command(
            "/retry",
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &default_path,
            &args,
        );
        match result {
            InteractiveAction::ProcessQuestion(q) => assert_eq!(q, "What is Rust?"),
            _ => panic!("expected ProcessQuestion"),
        }
    }

    #[test]
    fn test_interactive_command_save_and_load() {
        let tmp = tempfile::TempDir::new().unwrap();
        let session_path = tmp.path().join("test_session.json");

        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_user_message("hello");
        session.add_assistant_message("hi there");

        let mut sources = Vec::new();
        let mut last_q = None;
        let args = create_test_chat_args();

        let result = handle_interactive_command(
            &format!("/save {}", session_path.display()),
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &session_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Continue));
        assert!(session_path.exists());

        let mut session2 = ChatSession::new(ChatTemplate::ChatML);
        let result = handle_interactive_command(
            &format!("/load {}", session_path.display()),
            &mut session2,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &session_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Continue));
        assert_eq!(session2.len(), 2);
        assert_eq!(session2.turn_count(), 1);
    }

    #[test]
    fn test_interactive_command_help() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        let mut sources = Vec::new();
        let mut last_q = None;
        let default_path = PathBuf::from("/tmp/test_session.json");
        let args = create_test_chat_args();
        let result = handle_interactive_command(
            "/help",
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &default_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Continue));
    }

    #[test]
    fn test_interactive_command_unknown() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        let mut sources = Vec::new();
        let mut last_q = None;
        let default_path = PathBuf::from("/tmp/test_session.json");
        let args = create_test_chat_args();
        let result = handle_interactive_command(
            "/foobar",
            &mut session,
            &mut sources,
            &mut last_q,
            &ChatTemplate::ChatML,
            &default_path,
            &args,
        );
        assert!(matches!(result, InteractiveAction::Continue));
    }

    #[test]
    fn test_interactive_history_path() {
        let path = interactive_history_path();
        if let Some(p) = path {
            assert!(p.to_string_lossy().contains(".chatvcode"));
            assert!(p.to_string_lossy().contains("history"));
        }
    }

    #[test]
    fn test_interactive_default_session_path() {
        let path = interactive_default_session_path();
        assert!(path.to_string_lossy().contains("session.json"));
    }

    #[test]
    fn test_display_interactive_sources_empty() {
        display_interactive_sources(&[]);
    }

    #[test]
    fn test_display_interactive_sources_with_data() {
        let sources = vec![SourceReference {
            chunk_id: "id1".to_string(),
            file_path: PathBuf::from("src/main.rs"),
            kind: chatvcode_core::ChunkKind::Function,
            symbol_name: Some("main".to_string()),
            start_line: 10,
            end_line: 20,
            score: 0.95,
            snippet: "fn main() {}".to_string(),
        }];
        display_interactive_sources(&sources);
    }

    #[test]
    fn test_session_context_token_limit_accessor() {
        let session = ChatSession::new(ChatTemplate::ChatML)
            .max_context_tokens(4096)
            .reserve_for_response(512);
        assert_eq!(session.context_token_limit(), 4096);
        assert_eq!(session.response_token_reserve(), 512);
    }
}
