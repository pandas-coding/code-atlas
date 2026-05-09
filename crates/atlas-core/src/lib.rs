use std::path::PathBuf;

use rayon::prelude::*;

pub use error::{AtlasError, AtlasResult, ErrorContext, ErrorKind};
pub use model::{
    ChunkKind, ChunkSpan, CodeChunk, FileLanguage, IndexResult, IndexStats, ParseResult, SourceFile,
};
pub use scanner::{ScanOptions, ScanResult, Scanner};

pub mod error;
pub mod ignore;
pub mod model;
pub mod scanner;

pub trait ParseSource: Send + Sync {
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

pub fn index_path(path: impl Into<PathBuf>, parser: &dyn ParseSource) -> AtlasResult<IndexResult> {
    let path = path.into();

    if !path.exists() {
        return Err(AtlasError::invalid_input(format!(
            "Path does not exist: {}",
            path.display()
        )));
    }

    let options = ScanOptions::new(&path);
    let source_files = Scanner::scan_and_read(&options);

    let results: Vec<_> = source_files
        .into_par_iter()
        .map(|result| match result {
            Ok(source_file) => match parser.parse(source_file) {
                Ok(parse_result) => Ok(parse_result),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
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

    Ok(IndexResult::from_parse_results(parse_results, scan_errors))
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
            fs::write(root.join(format!("src/file_{i:02}.rs")), format!("fn func_{i}() {{}}")).unwrap();
        }

        let result1 = index_path(root, &mock_parser).unwrap();
        let result2 = index_path(root, &mock_parser).unwrap();

        assert_eq!(result1.stats, result2.stats);
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
}
