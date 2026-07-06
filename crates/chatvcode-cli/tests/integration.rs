use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

use chatvcode_cli::chatvcode_core::{
    ChatOptions, ChatResponse, ChunkKind, ErrorKind, ErrorSeverity, FileLanguage, SearchOptions,
    SourceReference, index_path, search,
};
use chatvcode_cli::chatvcode_llm::{
    ChatPromptBuilder, ChatTemplate, GenerationParams, LlmService, MockLlmService, StopReason,
    StreamEvent, TokenUsage,
};
use chatvcode_cli::chatvcode_parser::parse_source;
use chatvcode_cli::chatvcode_vdb::EmbeddingConfig;
use chatvcode_cli::{Cli, Commands, format_index_result, format_search_results};
use chatvcode_cli::agent::AgentCommand;
use clap::Parser;

fn create_rust_project() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() { println!(\"hello\"); }").unwrap();
    fs::write(
        root.join("src/lib.rs"),
        "pub struct Point { x: f64, y: f64 }\n\npub fn new_point(x: f64, y: f64) -> Point { Point { x, y } }",
    )
    .unwrap();

    tmp
}

fn create_mixed_project() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
    fs::write(root.join("src/index.js"), "function greet() { return 'hi'; }").unwrap();
    fs::write(root.join("src/app.tsx"), "export function App() { return <div />; }").unwrap();

    fs::create_dir_all(root.join("target/debug")).unwrap();
    fs::write(root.join("target/debug/program"), "binary").unwrap();

    fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
    fs::write(root.join("node_modules/pkg/index.js"), "module.exports = {};").unwrap();

    fs::create_dir_all(root.join(".git/objects")).unwrap();
    fs::write(root.join(".git/config"), "[core]").unwrap();

    fs::write(root.join("image.png"), "fake-image").unwrap();
    fs::write(root.join("Cargo.lock"), "").unwrap();

    tmp
}

#[test]
fn cli_index_directory_scans_rust_project() {
    let tmp = create_rust_project();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 2);
    assert_eq!(result.stats.parsed_files, 2);
    assert!(result.stats.total_errors == 0);
}

#[test]
fn cli_index_directory_extracts_chunks() {
    let tmp = create_rust_project();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert!(result.stats.total_chunks >= 3);

    let kinds: Vec<_> = result
        .files
        .iter()
        .flat_map(|f| f.chunks.iter().map(|c| c.kind))
        .collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Struct));
}

#[test]
fn cli_index_mixed_project() {
    let tmp = create_mixed_project();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 4);
    assert_eq!(result.stats.parsed_files, 4);

    let languages: Vec<_> = result.files.iter().map(|f| f.file.language).collect();
    assert!(languages.contains(&FileLanguage::Rust));
    assert!(languages.contains(&FileLanguage::JavaScript));
    assert!(languages.contains(&FileLanguage::Tsx));
}

#[test]
fn cli_index_ignores_target_and_node_modules() {
    let tmp = create_mixed_project();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    for file in &result.files {
        let path_str = file.file.path.to_string_lossy();
        assert!(!path_str.contains("target"), "should not include target: {path_str}");
        assert!(!path_str.contains("node_modules"), "should not include node_modules: {path_str}");
        assert!(!path_str.contains(".git"), "should not include .git: {path_str}");
    }
}

#[test]
fn cli_index_nonexistent_path_returns_error() {
    let result = index_path("/nonexistent/path/that/does/not/exist", &parse_source);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind, ErrorKind::InvalidInput);
}

#[test]
fn cli_index_single_rust_file() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("test.rs");
    fs::write(&file_path, "fn test_fn() {}").unwrap();

    let result = index_path(&file_path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert_eq!(result.stats.total_chunks, 1);
    assert_eq!(result.files[0].chunks[0].kind, ChunkKind::Function);
    assert_eq!(result.files[0].chunks[0].symbol_name.as_deref(), Some("test_fn"));
}

#[test]
fn cli_index_single_js_file() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("app.js");
    fs::write(&file_path, "function hello() { return 42; }").unwrap();

    let result = index_path(&file_path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert_eq!(result.files[0].file.language, FileLanguage::JavaScript);
}

#[test]
fn cli_index_empty_directory() {
    let tmp = TempDir::new().unwrap();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 0);
    assert_eq!(result.stats.parsed_files, 0);
    assert_eq!(result.stats.total_chunks, 0);
}

#[test]
fn cli_index_directory_with_only_non_source_files() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("image.png"), "fake").unwrap();
    fs::write(tmp.path().join("data.pdf"), "fake").unwrap();
    fs::write(tmp.path().join(".hidden"), "fake").unwrap();

    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 0);
    assert_eq!(result.stats.parsed_files, 0);
}

#[test]
fn cli_format_output_contains_stats() {
    let tmp = create_rust_project();
    let result = index_path(tmp.path(), &parse_source).unwrap();
    let output = format_index_result(&result);

    assert!(output.contains("Indexing complete."));
    assert!(output.contains("Files scanned"));
    assert!(output.contains("Files parsed"));
    assert!(output.contains("Total chunks"));
    assert!(output.contains("Errors"));
}

#[test]
fn cli_format_output_shows_no_errors_when_clean() {
    let tmp = create_rust_project();
    let result = index_path(tmp.path(), &parse_source).unwrap();
    let output = format_index_result(&result);

    assert!(!output.contains("Failed files:"));
    assert!(!output.contains("Parse warnings:"));
}

#[test]
fn cli_format_output_shows_failed_files() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/good.rs"), "fn good() {}").unwrap();
    fs::write(root.join("src/bad.rs"), "#[cfg(not_exists)]\n#[cfg(not_exists2)]\nfn broken(()")
        .unwrap();

    let result = index_path(root, &parse_source).unwrap();
    let output = format_index_result(&result);

    assert!(output.contains("Indexing complete."));
}

#[test]
fn cli_index_rust_chunk_symbol_names() {
    let tmp = TempDir::new().unwrap();
    let code = r"
struct Point { x: f64, y: f64 }

fn new_point(x: f64, y: f64) -> Point { Point { x, y } }

enum Shape { Circle, Rectangle }
";
    fs::write(tmp.path().join("lib.rs"), code).unwrap();

    let result = index_path(tmp.path(), &parse_source).unwrap();

    let names: Vec<_> = result.files[0]
        .chunks
        .iter()
        .map(|c| c.symbol_name.as_deref().unwrap_or("<none>"))
        .collect();

    assert!(names.contains(&"Point"));
    assert!(names.contains(&"new_point"));
    assert!(names.contains(&"Shape"));
}

#[test]
fn cli_index_js_chunk_symbol_names() {
    let tmp = TempDir::new().unwrap();
    let code = r#"
class Animal {
    constructor(name) { this.name = name; }
}

function greet(name) { return "hello " + name; }
"#;
    fs::write(tmp.path().join("app.js"), code).unwrap();

    let result = index_path(tmp.path(), &parse_source).unwrap();

    let names: Vec<_> = result.files[0]
        .chunks
        .iter()
        .map(|c| c.symbol_name.as_deref().unwrap_or("<none>"))
        .collect();

    assert!(names.contains(&"Animal"));
    assert!(names.contains(&"greet"));
}

#[test]
fn cli_index_ts_interface_and_type() {
    let tmp = TempDir::new().unwrap();
    let code = r"
interface User { name: string; age: number; }
type ID = string | number;
";
    fs::write(tmp.path().join("types.ts"), code).unwrap();

    let result = index_path(tmp.path(), &parse_source).unwrap();

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();

    assert!(kinds.contains(&ChunkKind::Interface));
    assert!(kinds.contains(&ChunkKind::TypeAlias));
}

#[test]
fn cli_commands_index_parses_path() {
    let cli = Cli::try_parse_from(["chatvcode", "index", "/some/path"]);
    assert!(cli.is_ok());

    match cli.unwrap().command {
        Commands::Index { path, .. } => assert_eq!(path, "/some/path"),
        Commands::Search { .. } => {}
        Commands::Chat { .. } => {}
        Commands::Model { .. } => {},
        Commands::Agent(_) => {},
    }
}

#[test]
fn cli_consistent_results_across_runs() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    for i in 0..10 {
        fs::write(root.join(format!("src/file_{i:02}.rs")), format!("fn func_{i}() {{}}")).unwrap();
    }

    let result1 = index_path(root, &parse_source).unwrap();
    let result2 = index_path(root, &parse_source).unwrap();

    assert_eq!(result1.stats.total_files, result2.stats.total_files);
    assert_eq!(result1.stats.parsed_files, result2.stats.parsed_files);
    assert_eq!(result1.stats.total_chunks, result2.stats.total_chunks);
    assert_eq!(result1.stats.total_errors, result2.stats.total_errors);
    assert_eq!(result1.stats.files_by_language, result2.stats.files_by_language);
    assert_eq!(result1.stats.chunks_by_kind, result2.stats.chunks_by_kind);
}

#[test]
fn cli_index_nonexistent_path_is_unrecoverable() {
    let result = index_path("/nonexistent/path/that/does/not/exist", &parse_source);
    let err = result.unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidInput);
    assert_eq!(err.severity, ErrorSeverity::Unrecoverable);
    assert!(!err.is_recoverable());
}

#[test]
fn cli_format_output_distinguishes_error_severity() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/good.rs"), "fn good() {}").unwrap();
    fs::write(root.join("src/bad.rs"), "#[cfg(not_exists)]\n#[cfg(not_exists2)]\nfn broken(()")
        .unwrap();

    let result = index_path(root, &parse_source).unwrap();
    let output = format_index_result(&result);

    assert!(output.contains("Indexing complete."));
}

#[test]
fn cli_commands_search_parses_query() {
    let cli = Cli::try_parse_from([
        "chatvcode",
        "search",
        "find function",
        "--embedding-model",
        "/model.onnx",
    ]);
    assert!(cli.is_ok());

    match cli.unwrap().command {
        Commands::Search { query, .. } => assert_eq!(query, "find function"),
        Commands::Index { .. } => panic!("expected Search command"),
        Commands::Chat { .. } => panic!("expected Search command"),
        Commands::Model { .. } => panic!("expected Search command"),
        Commands::Agent(_) => panic!("expected Search command"),
    }
}

#[test]
fn cli_commands_search_default_top_k() {
    let cli =
        Cli::try_parse_from(["chatvcode", "search", "test", "--embedding-model", "/model.onnx"]);
    assert!(cli.is_ok());

    match cli.unwrap().command {
        Commands::Search { top_k, .. } => assert_eq!(top_k, 10),
        Commands::Index { .. } => panic!("expected Search command"),
        Commands::Chat { .. } => panic!("expected Search command"),
        Commands::Model { .. } => panic!("expected Search command"),
        Commands::Agent(_) => panic!("expected Search command"),
    }
}

#[test]
fn cli_commands_search_custom_top_k() {
    let cli = Cli::try_parse_from([
        "chatvcode",
        "search",
        "test",
        "--embedding-model",
        "/model.onnx",
        "--top-k",
        "5",
    ]);
    assert!(cli.is_ok());

    match cli.unwrap().command {
        Commands::Search { top_k, .. } => assert_eq!(top_k, 5),
        Commands::Index { .. } => panic!("expected Search command"),
        Commands::Chat { .. } => panic!("expected Search command"),
        Commands::Model { .. } => panic!("expected Search command"),
        Commands::Agent(_) => panic!("expected Search command"),
    }
}

#[test]
fn cli_commands_search_with_min_score() {
    let cli = Cli::try_parse_from([
        "chatvcode",
        "search",
        "test",
        "--embedding-model",
        "/model.onnx",
        "--min-score",
        "0.5",
    ]);
    assert!(cli.is_ok());

    match cli.unwrap().command {
        Commands::Search { min_score, .. } => assert_eq!(min_score, Some(0.5f32)),
        Commands::Index { .. } => panic!("expected Search command"),
        Commands::Chat { .. } => panic!("expected Search command"),
        Commands::Model { .. } => panic!("expected Search command"),
        Commands::Agent(_) => panic!("expected Search command"),
    }
}

#[test]
fn cli_search_without_embedding_model_returns_error() {
    use assert_cmd::Command;

    let mut cmd = Command::cargo_bin("chatvcode").unwrap();
    cmd.arg("search").arg("test query");
    cmd.assert().failure();
}

#[test]
fn cli_search_nonexistent_path_returns_error() {
    let config = EmbeddingConfig::new("/nonexistent/model.onnx", 384, 512);
    let opts = SearchOptions::new(config, "/nonexistent/vectors.vdb");

    let result = search("test query", "/nonexistent/path", &parse_source, &opts);
    assert!(result.is_err());
}

#[test]
fn cli_search_missing_vector_store_returns_error() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("main.rs"), "fn main() {}").unwrap();

    let config = EmbeddingConfig::new("/nonexistent/model.onnx", 384, 512);
    let vdb_path = tmp.path().join(".chatvcode").join("vectors.vdb");
    let opts = SearchOptions::new(config, vdb_path);

    let result = search("test query", tmp.path(), &parse_source, &opts);
    assert!(result.is_err());
}

#[test]
fn cli_format_search_results_empty() {
    let output = format_search_results("test query", &[]);
    assert!(output.contains("No results found"));
    assert!(output.contains("test query"));
}

#[test]
fn cli_format_search_results_with_data() {
    use chatvcode_cli::chatvcode_core::{ChunkSpan, CodeChunk, SearchResult};
    use std::path::PathBuf;

    let chunk = CodeChunk {
        id: "test.rs:function:main:0".to_string(),
        file_path: PathBuf::from("src/main.rs"),
        language: FileLanguage::Rust,
        kind: ChunkKind::Function,
        symbol_name: Some("main".to_string()),
        span: ChunkSpan::new(0, 20, 0, 2),
        source_text: "fn main() {\n    println!(\"hello\");\n}".to_string(),
    };
    let result = SearchResult { chunk_id: chunk.id.clone(), score: 0.95, chunk };

    let output = format_search_results("find main", &[result]);
    assert!(output.contains("find main"));
    assert!(output.contains("0.9500"));
    assert!(output.contains("main.rs"));
    assert!(output.contains("function"));
    assert!(output.contains("main"));
    assert!(output.contains("fn main()"));
}

// ===========================================================================
// Chat command integration tests
// ===========================================================================

#[test]
fn cli_chat_command_parses_question() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "What is this codebase?"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { question, .. } => {
            assert_eq!(question, "What is this codebase?");
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_default_path() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { path, .. } => {
            assert_eq!(path, ".");
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_with_path() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--path", "/my/project"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { question, path, .. } => {
            assert_eq!(question, "test");
            assert_eq!(path, "/my/project");
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_with_model() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--model", "/models/qwen.gguf"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { model, .. } => {
            assert_eq!(model, Some("/models/qwen.gguf".to_string()));
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_with_temperature() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--temperature", "0.3"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { temperature, .. } => {
            assert!((temperature - 0.3).abs() < f32::EPSILON);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_with_max_tokens() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--max-tokens", "2048"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { max_tokens, .. } => {
            assert_eq!(max_tokens, 2048);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_with_template() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--template", "chatml"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { template, .. } => {
            assert_eq!(template, "chatml");
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_with_system_prompt() {
    let cli = Cli::try_parse_from([
        "chatvcode",
        "chat",
        "test",
        "--system-prompt",
        "You are a Rust expert.",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { system_prompt, .. } => {
            assert_eq!(system_prompt, Some("You are a Rust expert.".to_string()));
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_stream_flag() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--stream=false"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { stream, .. } => {
            assert!(!stream);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_stream_default_is_true() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { stream, .. } => {
            assert!(stream); // default should be true
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_stream_explicit_true() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--stream=true"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { stream, .. } => {
            assert!(stream);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_stream_bare_flag() {
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
fn cli_chat_command_retrieval_default_is_true() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { retrieval, .. } => {
            assert!(retrieval); // default should be true
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_retrieval_disabled() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--retrieval=false"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { retrieval, .. } => {
            assert!(!retrieval); // explicitly disabled
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_retrieval_bare_flag() {
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
fn cli_chat_command_json_flag() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--json"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { json, .. } => {
            assert!(json);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_n_ctx() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--n-ctx", "8192"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { n_ctx, .. } => {
            assert_eq!(n_ctx, 8192);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_n_gpu_layers() {
    let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--n-gpu-layers=-1"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Commands::Chat { n_gpu_layers, .. } => {
            assert_eq!(n_gpu_layers, -1);
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_command_all_options() {
    let cli = Cli::try_parse_from([
        "chatvcode",
        "chat",
        "Explain the code",
        "--path",
        "/tmp/project",
        "--model",
        "/models/test.gguf",
        "--temperature",
        "0.5",
        "--max-tokens",
        "1024",
        "--top-k",
        "50",
        "--top-p",
        "0.95",
        "--template",
        "llama3",
        "--system-prompt",
        "Be concise",
        "--stream=false",
        "--json",
        "--n-ctx",
        "4096",
        "--n-gpu-layers",
        "0",
        "--embedding-model",
        "/models/embed.onnx",
        "--top-k-retrieval",
        "5",
        "--min-score",
        "0.6",
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
            embedding_model,
            top_k_retrieval,
            min_score,
            ..
        } => {
            assert_eq!(question, "Explain the code");
            assert_eq!(path, "/tmp/project");
            assert_eq!(model, Some("/models/test.gguf".to_string()));
            assert!((temperature - 0.5).abs() < f32::EPSILON);
            assert_eq!(max_tokens, 1024);
            assert_eq!(top_k, 50);
            assert!((top_p - 0.95).abs() < f32::EPSILON);
            assert_eq!(template, "llama3");
            assert_eq!(system_prompt, Some("Be concise".to_string()));
            assert!(!stream);
            assert!(json);
            assert_eq!(n_ctx, 4096);
            assert_eq!(n_gpu_layers, 0);
            assert_eq!(embedding_model, Some("/models/embed.onnx".to_string()));
            assert_eq!(top_k_retrieval, 5);
            assert_eq!(min_score, Some(0.6));
        }
        _ => panic!("expected Chat command"),
    }
}

#[test]
fn cli_chat_with_mock_llm_inference() {
    // Test that MockLlmService can perform inference like a real model
    let service = MockLlmService::new("The function parses configuration files.");
    let params = GenerationParams::default();
    let cancel = std::sync::atomic::AtomicBool::new(false);

    let response = service
        .infer("What does this function do?", &params, Some(&cancel))
        .unwrap();

    assert_eq!(response.text, "The function parses configuration files.");
    assert_eq!(response.stop_reason, StopReason::Eos);
    assert!(response.token_usage.prompt_tokens > 0);
    assert!(response.token_usage.completion_tokens > 0);
}

#[test]
fn cli_chat_with_mock_llm_streaming() {
    // Test that MockLlmService can stream tokens
    let service = MockLlmService::new("Hello world from streaming");
    let params = GenerationParams::default();
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let rx = service.infer_stream("test", &params, Some(cancel)).unwrap();

    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
        events.push(event);
    }

    assert!(!events.is_empty());
    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert_eq!(events.last(), Some(&StreamEvent::Completed));

    let full_text: String = events
        .iter()
        .filter_map(|e| e.as_token().map(String::from))
        .collect();
    assert_eq!(full_text, "Hello world from streaming");
}

#[test]
fn cli_chat_mock_llm_cancelled() {
    let service = MockLlmService::new("Should not see this");
    let params = GenerationParams::default();
    let cancel = std::sync::atomic::AtomicBool::new(true); // Pre-cancelled

    let response = service.infer("test", &params, Some(&cancel)).unwrap();
    assert_eq!(response.stop_reason, StopReason::Cancelled);
    assert!(response.text.is_empty());
}

#[test]
fn cli_chat_prompt_builder_with_rag_context() {
    // Verify that prompt builder produces correct output with RAG context
    let prompt = ChatPromptBuilder::new(ChatTemplate::ChatML)
        .system_prompt("You are a helpful coding assistant.")
        .user_question("What does the parse function do?")
        .context("fn parse(input: &str) -> Result<Config> { ... }")
        .build()
        .unwrap();

    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.contains("What does the parse function do?"));
    assert!(prompt.contains("parse"));
    assert!(prompt.contains("[Retrieved Context]"));
}

#[test]
fn cli_chat_prompt_builder_without_context() {
    // Verify that prompt builder works without context (no RAG)
    let prompt = ChatPromptBuilder::new(ChatTemplate::ChatML)
        .system_prompt("You are helpful.")
        .user_question("What is Rust?")
        .build()
        .unwrap();

    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("What is Rust?"));
    assert!(!prompt.contains("[Retrieved Context]"));
}

#[test]
fn cli_chat_source_reference_display_path() {
    let source = SourceReference {
        chunk_id: "id1".to_string(),
        file_path: PathBuf::from("src/parser.rs"),
        kind: ChunkKind::Function,
        symbol_name: Some("parse_config".to_string()),
        start_line: 42,
        end_line: 58,
        score: 0.91,
        snippet: "fn parse_config() { ... }".to_string(),
    };

    assert_eq!(source.display_path(), "src/parser.rs:42:parse_config-58");
}

#[test]
fn cli_chat_source_reference_display_path_no_symbol() {
    let source = SourceReference {
        chunk_id: "id2".to_string(),
        file_path: PathBuf::from("src/utils.rs"),
        kind: ChunkKind::Function,
        symbol_name: None,
        start_line: 10,
        end_line: 20,
        score: 0.85,
        snippet: "some code".to_string(),
    };

    assert_eq!(source.display_path(), "src/utils.rs:10-20");
}

#[test]
fn cli_chat_response_no_context_flag() {
    let response_with_sources = ChatResponse {
        answer: "Answer".to_string(),
        sources: vec![SourceReference {
            chunk_id: "id".to_string(),
            file_path: PathBuf::from("src/main.rs"),
            kind: ChunkKind::Function,
            symbol_name: None,
            start_line: 1,
            end_line: 5,
            score: 0.9,
            snippet: "code".to_string(),
        }],
        token_usage: TokenUsage::new(10, 5),
        stop_reason: StopReason::Eos,
        duration: std::time::Duration::from_millis(100),
        search_duration: std::time::Duration::from_millis(10),
        inference_duration: std::time::Duration::from_millis(90),
        retrieved_count: 1,
        used_count: 1,
    };
    assert!(!response_with_sources.is_no_context());

    let response_without_sources = ChatResponse {
        answer: "No context answer".to_string(),
        sources: vec![],
        token_usage: TokenUsage::new(10, 5),
        stop_reason: StopReason::Eos,
        duration: std::time::Duration::from_millis(100),
        search_duration: std::time::Duration::from_millis(10),
        inference_duration: std::time::Duration::from_millis(90),
        retrieved_count: 0,
        used_count: 0,
    };
    assert!(response_without_sources.is_no_context());
}

#[test]
fn cli_chat_response_format_sources() {
    let response = ChatResponse {
        answer: "Answer".to_string(),
        sources: vec![
            SourceReference {
                chunk_id: "id1".to_string(),
                file_path: PathBuf::from("src/main.rs"),
                kind: ChunkKind::Function,
                symbol_name: Some("main".to_string()),
                start_line: 10,
                end_line: 20,
                score: 0.95,
                snippet: "fn main() {}".to_string(),
            },
            SourceReference {
                chunk_id: "id2".to_string(),
                file_path: PathBuf::from("src/lib.rs"),
                kind: ChunkKind::Struct,
                symbol_name: Some("Config".to_string()),
                start_line: 5,
                end_line: 15,
                score: 0.82,
                snippet: "struct Config {}".to_string(),
            },
        ],
        token_usage: TokenUsage::new(50, 20),
        stop_reason: StopReason::Eos,
        duration: std::time::Duration::from_millis(200),
        search_duration: std::time::Duration::from_millis(20),
        inference_duration: std::time::Duration::from_millis(180),
        retrieved_count: 2,
        used_count: 2,
    };

    let formatted = response.format_sources();
    assert!(formatted.contains("Sources:"));
    assert!(formatted.contains("src/main.rs"));
    assert!(formatted.contains("src/lib.rs"));
    assert!(formatted.contains("0.950"));
    assert!(formatted.contains("0.820"));
}

#[test]
fn cli_chat_generation_params_from_options() {
    // Verify GenerationParams can be built from CLI options
    let params = GenerationParams::default()
        .with_temperature(0.5)
        .with_top_p(0.95)
        .with_top_k(50)
        .with_max_tokens(1024);

    assert!((params.temperature - 0.5).abs() < f32::EPSILON);
    assert!((params.top_p - 0.95).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 50);
    assert_eq!(params.max_tokens, 1024);
}

#[test]
fn cli_chat_chat_options_builder() {
    let options = ChatOptions::new("/tmp/project")
        .with_top_k(5)
        .with_min_score(0.7)
        .with_context_token_budget(2048)
        .system_prompt("You are a Rust expert.")
        .with_chat_template(ChatTemplate::ChatML)
        .with_generation_params(GenerationParams::default().with_max_tokens(256));

    assert_eq!(options.top_k, 5);
    assert_eq!(options.min_score, Some(0.7));
    assert_eq!(options.context_token_budget, 2048);
    assert_eq!(options.system_prompt.as_deref(), Some("You are a Rust expert."));
    assert_eq!(options.chat_template, ChatTemplate::ChatML);
    assert_eq!(options.generation_params.max_tokens, 256);
}

#[test]
fn cli_chat_chat_template_variants() {
    // Verify all template variants are parseable from CLI
    let templates = ["auto", "raw", "chatml", "llama3"];
    for tmpl in templates {
        let cli = Cli::try_parse_from(["chatvcode", "chat", "test", "--template", tmpl]);
        assert!(cli.is_ok(), "Failed to parse --template {tmpl}");
    }
}

#[test]
fn cli_chat_mock_llm_custom_response() {
    let service = MockLlmService::new("Custom response text for testing.");
    let params = GenerationParams::default();
    let response = service.infer("any prompt", &params, None).unwrap();
    assert_eq!(response.text, "Custom response text for testing.");
}

#[test]
fn cli_chat_mock_llm_max_tokens_truncation() {
    let service = MockLlmService::new("This is a longer response that should be truncated.");
    let params = GenerationParams { max_tokens: 3, ..GenerationParams::default() };
    let response = service.infer("test", &params, None).unwrap();
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
}

#[test]
fn cli_chat_chat_prompt_is_missing_question_error() {
    // ChatPromptBuilder requires a user question
    let result = ChatPromptBuilder::new(ChatTemplate::ChatML)
        .system_prompt("You are helpful.")
        .build();
    assert!(result.is_err());
}

#[test]
fn cli_chat_format_sources_display_with_data() {
    let sources = vec![SourceReference {
        chunk_id: "id1".to_string(),
        file_path: PathBuf::from("src/main.rs"),
        kind: ChunkKind::Function,
        symbol_name: Some("main".to_string()),
        start_line: 10,
        end_line: 20,
        score: 0.95,
        snippet: "fn main() {}".to_string(),
    }];
    let output = format!("{:#?}", sources);
    assert!(output.contains("src/main.rs"));
    assert!(output.contains("main"));
}

#[test]
fn cli_chat_rag_prompt_format_snippets() {
    let meta = chatvcode_cli::chatvcode_core::ChunkMetadata {
        chunk_id: "test_id".to_string(),
        file_path: PathBuf::from("src/parser.rs"),
        language: "rust".to_string(),
        kind: ChunkKind::Function,
        symbol_name: Some("parse".to_string()),
        start_line: 42,
        end_line: 58,
        start_byte: 500,
        end_byte: 800,
        source_text: "fn parse() {}".to_string(),
    };
    let snippets = chatvcode_cli::chatvcode_core::format_context_snippets(&[(meta, 0.89)]);
    assert_eq!(snippets.len(), 1);
    assert!(snippets[0].contains("src/parser.rs:42-58"));
    assert!(snippets[0].contains("function: parse"));
    assert!(snippets[0].contains("0.890"));
}

// ===========================================================================
// Agent command integration tests
// ===========================================================================

#[test]
fn cli_agent_command_parses_question_and_path() {
    let cli = Cli::try_parse_from([
        "chatvcode", "agent", "How does error handling work?", "--path", "/my/project",
    ]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { question, path, .. }) = parsed.command {
            assert_eq!(question.as_deref(), Some("How does error handling work?"));
            assert_eq!(path, "/my/project");
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_default_path_is_dot() {
    let cli = Cli::try_parse_from(["chatvcode", "agent", "test"]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { path, .. }) = parsed.command {
            assert_eq!(path, ".");
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_interactive_short_flag() {
    let cli = Cli::try_parse_from(["chatvcode", "agent", "-i", "--path", "/tmp/p"]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { interactive, path, question, .. }) = parsed.command {
            assert!(interactive);
            assert_eq!(path, "/tmp/p");
            assert!(question.is_none());
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_verbose_short_flag() {
    let cli = Cli::try_parse_from(["chatvcode", "agent", "hi", "-v"]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { verbose, .. }) = parsed.command {
            assert!(verbose);
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_json_and_no_stream() {
    let cli = Cli::try_parse_from([
        "chatvcode", "agent", "hi", "--json", "--no-stream", "--max-steps", "8", "--timeout", "60",
    ]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { json, no_stream, max_steps, timeout, .. }) = parsed.command {
            assert!(json);
            assert!(no_stream);
            assert_eq!(max_steps, Some(8));
            assert_eq!(timeout, Some(60));
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_tools_csv() {
    let cli = Cli::try_parse_from([
        "chatvcode", "agent", "hi", "--tools", "read_file,grep_code",
    ]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { tools, .. }) = parsed.command {
            assert_eq!(tools.as_deref(), Some("read_file,grep_code"));
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_confirm_plan_and_model() {
    let cli = Cli::try_parse_from([
        "chatvcode", "agent", "hi", "--confirm-plan", "--model", "/m/qwen.gguf",
    ]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { confirm_plan, model, .. }) = parsed.command {
            assert!(confirm_plan);
            assert_eq!(model.as_deref(), Some("/m/qwen.gguf"));
        } else {
            panic!("expected Agent command");
        }
    }
}

#[test]
fn cli_agent_command_mock_llm() {
    let cli = Cli::try_parse_from([
        "chatvcode", "agent", "hi", "--mock-llm", "--mock-llm-response", "fake answer",
    ]);
    assert!(cli.is_ok());
    if let Ok(parsed) = cli {
        if let Commands::Agent(AgentCommand { mock_llm, mock_llm_response, .. }) = parsed.command {
            assert!(mock_llm);
            assert_eq!(mock_llm_response.as_deref(), Some("fake answer"));
        } else {
            panic!("expected Agent command");
        }
    }
}
