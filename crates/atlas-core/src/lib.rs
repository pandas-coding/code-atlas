use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

pub use error::{AtlasError, AtlasResult, ErrorContext, ErrorKind, ErrorSeverity};
pub use model::{
    ChunkKind, ChunkSpan, CodeChunk, EmbeddingOptions, FileLanguage, FileState, IndexOptions,
    IndexResult, IndexState, IndexStats, ParseResult, SourceFile,
};
pub use scanner::{ScanOptions, ScanResult, Scanner};

pub mod error;
pub mod ignore;
pub mod incremental;
pub mod model;
pub mod scanner;

/// Trait for parsing source files into [`ParseResult`]s.
///
/// This abstraction decouples `atlas-core` from `atlas-parser`, allowing
/// the parser implementation to be injected (e.g., for testing or future
/// parser swaps).
pub trait ParseSource: Send + Sync {
    /// Parses a single source file and returns the result.
    fn parse(&self, source_file: SourceFile) -> AtlasResult<ParseResult>;
}

impl<F> ParseSource for F
where
    F: Fn(SourceFile) -> AtlasResult<ParseResult> + Send + Sync,
{
    fn parse(&self, source_file: SourceFile) -> AtlasResult<ParseResult> {
        self(source_file)
    }
}

/// Indexes all source files under the given path with default options.
///
/// Convenience wrapper around [`index_path_with_options`].
pub fn index_path(path: impl Into<PathBuf>, parser: &dyn ParseSource) -> AtlasResult<IndexResult> {
    index_path_with_options(path, parser, &IndexOptions::default())
}

/// Indexes all source files under the given path with the provided options.
///
/// This is the main entry point for the indexing pipeline. It performs:
/// 1. Directory scanning and file discovery (with ignore rules)
/// 2. Incremental state loading (if configured)
/// 3. Parallel file reading (with large-file handling)
/// 4. Parallel parsing via the provided `parser`
/// 5. Chunk splitting for oversized chunks
/// 6. Result aggregation, statistics computation, and state persistence
/// 7. Embedding generation and vector store persistence (if configured)
///
/// Returns an [`IndexResult`] with all parsed chunks and statistics,
/// or an [`AtlasError`] if the path is invalid.
pub fn index_path_with_options(
    path: impl Into<PathBuf>,
    parser: &dyn ParseSource,
    options: &IndexOptions,
) -> AtlasResult<IndexResult> {
    let path = path.into();
    let start = Instant::now();

    if !path.exists() {
        let err = AtlasError::invalid_input(format!("Path does not exist: {}", path.display()));
        log::error!("{}", err);
        return Err(err);
    }

    log::info!("Starting index for path: {}", path.display());

    let mut scan_options = ScanOptions::new(&path);
    scan_options.large_file_threshold = Some(options.large_file_threshold);
    scan_options.large_file_max_lines = Some(options.large_file_max_lines);

    let incremental_state = if let Some(ref state_path) = options.incremental_state_path {
        match IndexState::load(state_path) {
            Ok(state) => {
                log::info!("Loaded incremental state from {}", state_path.display());
                Some(state)
            }
            Err(e) => {
                log::warn!("Could not load incremental state: {}", e);
                None
            }
        }
    } else {
        None
    };

    let source_files = Scanner::scan_and_read(&scan_options);

    let total_scanned = source_files.len();
    let scan_errors: Vec<_> = source_files.iter().filter(|r| r.is_err()).collect();
    log::info!(
        "Scan complete: {} source files found, {} scan errors",
        total_scanned - scan_errors.len(),
        scan_errors.len()
    );

    let files_to_parse: Vec<_> = if let Some(ref state) = incremental_state {
        source_files
            .into_iter()
            .filter(|result| match result {
                Ok(sf) => match std::fs::metadata(&sf.path) {
                    Ok(meta) => {
                        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                        let size = meta.len();
                        let changed = state.has_file_changed(&sf.path, mtime, size);
                        if !changed {
                            log::debug!("Skipping unchanged file: {}", sf.path.display());
                        }
                        changed
                    }
                    Err(_) => true,
                },
                Err(_) => true,
            })
            .collect()
    } else {
        source_files
    };

    let results: Vec<_> = files_to_parse
        .into_par_iter()
        .map(|result| match result {
            Ok(source_file) => {
                log::debug!("Parsing file: {}", source_file.path.display());
                match parser.parse(source_file) {
                    Ok(parse_result) => {
                        if !parse_result.errors.is_empty() {
                            log::warn!(
                                "Parse warnings in {}: {}",
                                parse_result.file.path.display(),
                                parse_result.errors.len()
                            );
                        }
                        Ok(parse_result)
                    }
                    Err(e) => {
                        log::error!("Failed to parse file: {}", e);
                        Err(e)
                    }
                }
            }
            Err(e) => {
                log::error!("Scan error: {}", e);
                Err(e)
            }
        })
        .collect();

    let mut parse_results = Vec::new();
    let mut scan_errors = Vec::new();

    for result in results {
        match result {
            Ok(parse_result) => parse_results.push(parse_result),
            Err(e) => scan_errors.push(e),
        }
    }

    if options.chunk_split_threshold > 0 {
        for parse_result in &mut parse_results {
            let threshold = options.chunk_split_threshold;
            let mut split_chunks = Vec::new();
            for chunk in parse_result.chunks.drain(..) {
                if chunk.source_text.len() > threshold {
                    let sub_chunks = split_large_chunk(&chunk, threshold);
                    split_chunks.extend(sub_chunks);
                } else {
                    split_chunks.push(chunk);
                }
            }
            parse_result.chunks = split_chunks;
        }
    }

    let mut index_result = IndexResult::from_parse_results(parse_results, scan_errors);
    let elapsed = start.elapsed();
    index_result.set_elapsed_ms(elapsed.as_millis() as u64);

    if let Some(ref state_path) = options.incremental_state_path {
        let mut new_state = incremental_state.unwrap_or_else(IndexState::new);
        for file_result in &index_result.files {
            let path = &file_result.file.path;
            if let Ok(meta) = std::fs::metadata(path) {
                let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                let size = meta.len();
                new_state.record_file(path, mtime, size, file_result.chunks.len());
            }
        }
        if let Err(e) = new_state.save(state_path) {
            log::warn!("Failed to save incremental state: {}", e);
        }
    }

    if let Some(ref embedding_opts) = options.embedding {
        let (embedded, emb_errors, dimension) =
            run_embedding(&index_result, embedding_opts);
        index_result.stats.embedded_chunks = embedded;
        index_result.stats.embedding_errors = emb_errors;
        index_result.stats.embedding_dimension = dimension;
    }

    log::info!(
        "Index complete: {} files parsed, {} chunks, {} errors in {}ms",
        index_result.stats.parsed_files,
        index_result.stats.total_chunks,
        index_result.stats.total_errors,
        index_result.stats.elapsed_ms
    );

    Ok(index_result)
}

fn run_embedding(
    index_result: &IndexResult,
    opts: &EmbeddingOptions,
) -> (usize, usize, usize) {
    use atlas_vdb::{EmbeddingService, EmbeddingVector, InMemoryVectorStore, VectorStore};

    log::info!("Starting embedding generation for {} chunks", index_result.stats.total_chunks);

    let embedding_service = match atlas_vdb::OnnxEmbeddingService::new(opts.config.clone()) {
        Ok(svc) => svc,
        Err(e) => {
            log::error!("Failed to initialize embedding service: {}", e);
            return (0, index_result.stats.total_chunks, 0);
        }
    };

    let all_chunks: Vec<&CodeChunk> = index_result
        .files
        .iter()
        .flat_map(|f| f.chunks.iter())
        .collect();

    if all_chunks.is_empty() {
        log::info!("No chunks to embed");
        return (0, 0, 0);
    }

    let batch_size = if opts.batch_size == 0 { 32 } else { opts.batch_size };
    let mut store = InMemoryVectorStore::new();
    let mut embedded = 0usize;
    let mut emb_errors = 0usize;

    for batch in all_chunks.chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|c| c.source_text.as_str()).collect();
        match embedding_service.embed(&texts) {
            Ok(vectors) => {
                let mut embedding_vectors = Vec::with_capacity(vectors.len());
                for (chunk, vector) in batch.iter().zip(vectors.into_iter()) {
                    embedding_vectors.push(EmbeddingVector::new(&chunk.id, vector));
                }
                match store.add(embedding_vectors) {
                    Ok(()) => embedded += batch.len(),
                    Err(e) => {
                        log::warn!("Failed to add embedding vectors to store: {}", e);
                        emb_errors += batch.len();
                    }
                }
            }
            Err(e) => {
                log::warn!("Embedding inference failed for batch: {}", e);
                emb_errors += batch.len();
            }
        }
    }

    let dimension = embedding_service.dimension();

    if embedded > 0 {
        if let Some(parent) = opts.vector_store_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::warn!("Failed to create vector store directory: {}", e);
            }
        }
        match store.save(&opts.vector_store_path) {
            Ok(()) => {
                log::info!(
                    "Saved vector store with {} vectors to {}",
                    store.len(),
                    opts.vector_store_path.display()
                );
            }
            Err(e) => {
                log::warn!("Failed to save vector store: {}", e);
            }
        }
    }

    log::info!(
        "Embedding complete: {} chunks embedded, {} errors, dimension {}",
        embedded,
        emb_errors,
        dimension
    );

    (embedded, emb_errors, dimension)
}

/// Splits a large chunk into smaller pieces by blank-line boundaries.
///
/// If the chunk cannot be meaningfully split (no blank lines), it is
/// returned as-is inside a single-element vector.
fn split_large_chunk(chunk: &CodeChunk, _threshold: usize) -> Vec<CodeChunk> {
    let lines: Vec<&str> = chunk.source_text.lines().collect();
    if lines.len() <= 1 {
        return vec![chunk.clone()];
    }

    let mut sub_chunks = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut current_start_line = chunk.span.start_line;
    let mut current_start_byte = chunk.span.start_byte;
    let line_offsets = line_byte_offsets(&chunk.source_text);

    for (i, line) in lines.iter().enumerate() {
        let is_blank = line.trim().is_empty();

        if is_blank && !current_lines.is_empty() {
            let text = current_lines.join("\n");
            if !text.is_empty() {
                let end_byte = current_start_byte + text.len();
                let end_line = current_start_line + current_lines.len().saturating_sub(1);
                sub_chunks.push(make_sub_chunk(
                    chunk,
                    current_start_byte,
                    end_byte,
                    current_start_line,
                    end_line,
                    &text,
                ));
            }
            current_start_line = chunk.span.start_line + i + 1;
            current_start_byte =
                if i + 1 < line_offsets.len() { line_offsets[i + 1] } else { chunk.span.end_byte };
            current_lines.clear();
        } else {
            current_lines.push(line);
        }
    }

    if !current_lines.is_empty() {
        let text = current_lines.join("\n");
        if !text.is_empty() {
            let end_byte = current_start_byte + text.len();
            let end_line = current_start_line + current_lines.len().saturating_sub(1);
            sub_chunks.push(make_sub_chunk(
                chunk,
                current_start_byte,
                end_byte,
                current_start_line,
                end_line,
                &text,
            ));
        }
    }

    if sub_chunks.is_empty() { vec![chunk.clone()] } else { sub_chunks }
}

fn make_sub_chunk(
    parent: &CodeChunk,
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
    text: &str,
) -> CodeChunk {
    CodeChunk {
        id: CodeChunk::generate_id(
            &parent.file_path,
            parent.kind,
            parent.symbol_name.as_deref(),
            start_line,
        ),
        file_path: parent.file_path.clone(),
        language: parent.language,
        kind: parent.kind,
        symbol_name: parent.symbol_name.clone(),
        span: ChunkSpan::new(start_byte, end_byte, start_line, end_line),
        source_text: text.to_string(),
    }
}

fn line_byte_offsets(text: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (i, c) in text.char_indices() {
        if c == '\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_project() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(root.join("src/index.js"), "function greet() {}").unwrap();

        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/program"), "binary").unwrap();

        tmp
    }

    fn mock_parser(source_file: SourceFile) -> AtlasResult<ParseResult> {
        let chunk = CodeChunk {
            id: CodeChunk::generate_id(&source_file.path, ChunkKind::Function, Some("mock"), 0),
            file_path: source_file.path.clone(),
            language: source_file.language,
            kind: ChunkKind::Function,
            symbol_name: Some("mock".to_string()),
            span: ChunkSpan::new(0, source_file.source_text.len(), 0, 0),
            source_text: source_file.source_text.clone(),
        };
        Ok(ParseResult::success(source_file, vec![chunk]))
    }

    #[test]
    fn test_index_path_scans_and_parses() {
        let tmp = create_test_project();
        let result = index_path(tmp.path(), &mock_parser).unwrap();

        assert_eq!(result.stats.total_files, 3);
        assert_eq!(result.stats.parsed_files, 3);
        assert_eq!(result.stats.total_chunks, 3);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_index_path_nonexistent_path() {
        let result = index_path("/nonexistent/path", &mock_parser);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn test_index_path_single_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.rs");
        fs::write(&file_path, "fn test() {}").unwrap();

        let result = index_path(&file_path, &mock_parser).unwrap();
        assert_eq!(result.stats.total_files, 1);
        assert_eq!(result.stats.parsed_files, 1);
        assert_eq!(result.stats.total_chunks, 1);
    }

    #[test]
    fn test_index_path_collects_parse_errors() {
        let tmp = create_test_project();

        let failing_parser = |_source_file: SourceFile| -> AtlasResult<ParseResult> {
            Err(AtlasError::parse("mock parse failure"))
        };

        let result = index_path(tmp.path(), &failing_parser).unwrap();
        assert_eq!(result.stats.parsed_files, 0);
        assert_eq!(result.stats.total_errors, 3);
    }

    #[test]
    fn test_index_path_skips_ignored_dirs() {
        let tmp = create_test_project();
        let result = index_path(tmp.path(), &mock_parser).unwrap();

        for file in &result.files {
            let path_str = file.file.path.to_string_lossy();
            assert!(!path_str.contains("target"));
        }
    }

    #[test]
    fn test_parallel_index_results_match_sequential() {
        let tmp = create_test_project();

        let result = index_path(tmp.path(), &mock_parser).unwrap();

        assert_eq!(result.stats.total_files, 3);
        assert_eq!(result.stats.parsed_files, 3);
        assert_eq!(result.stats.total_chunks, 3);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_parallel_index_consistent_results_across_runs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        for i in 0..20 {
            fs::write(root.join(format!("src/file_{i:02}.rs")), format!("fn func_{i}() {{}}"))
                .unwrap();
        }

        let result1 = index_path(root, &mock_parser).unwrap();
        let result2 = index_path(root, &mock_parser).unwrap();

        assert_eq!(result1.stats.total_files, result2.stats.total_files);
        assert_eq!(result1.stats.parsed_files, result2.stats.parsed_files);
        assert_eq!(result1.stats.skipped_files, result2.stats.skipped_files);
        assert_eq!(result1.stats.total_chunks, result2.stats.total_chunks);
        assert_eq!(result1.stats.total_errors, result2.stats.total_errors);
        assert_eq!(result1.stats.files_by_language, result2.stats.files_by_language);
        assert_eq!(result1.stats.chunks_by_language, result2.stats.chunks_by_language);
        assert_eq!(result1.stats.chunks_by_kind, result2.stats.chunks_by_kind);
        assert_eq!(result1.stats.total_source_bytes, result2.stats.total_source_bytes);
    }

    #[test]
    fn test_parallel_index_mixed_success_and_failure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/good.rs"), "fn good() {}").unwrap();
        fs::write(root.join("src/bad.rs"), "fn bad() {}").unwrap();

        let selective_parser = |source_file: SourceFile| -> AtlasResult<ParseResult> {
            if source_file.path.to_string_lossy().contains("bad") {
                Err(AtlasError::parse("selective failure"))
            } else {
                let chunk = CodeChunk {
                    id: CodeChunk::generate_id(
                        &source_file.path,
                        ChunkKind::Function,
                        Some("good"),
                        0,
                    ),
                    file_path: source_file.path.clone(),
                    language: source_file.language,
                    kind: ChunkKind::Function,
                    symbol_name: Some("good".to_string()),
                    span: ChunkSpan::new(0, source_file.source_text.len(), 0, 0),
                    source_text: source_file.source_text.clone(),
                };
                Ok(ParseResult::success(source_file, vec![chunk]))
            }
        };

        let result = index_path(root, &selective_parser).unwrap();
        assert_eq!(result.stats.parsed_files, 1);
        assert_eq!(result.stats.total_errors, 1);
    }

    #[test]
    fn test_index_path_nonexistent_path_is_unrecoverable() {
        let result = index_path("/nonexistent/path", &mock_parser);
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidInput);
        assert_eq!(err.severity, ErrorSeverity::Unrecoverable);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_parse_errors_are_recoverable() {
        let tmp = create_test_project();

        let failing_parser = |_source_file: SourceFile| -> AtlasResult<ParseResult> {
            Err(AtlasError::parse("mock parse failure"))
        };

        let result = index_path(tmp.path(), &failing_parser).unwrap();
        for err in &result.errors {
            assert!(err.is_recoverable());
        }
    }

    #[test]
    fn test_scan_errors_have_context() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

        let result = index_path(root, &mock_parser).unwrap();
        for err in &result.errors {
            assert!(err.context.operation.is_some() || err.context.path.is_some());
        }
    }
}
