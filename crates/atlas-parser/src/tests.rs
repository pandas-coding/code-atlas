use crate::*;
use atlas_core::{ErrorKind, FileLanguage, SourceFile};
use std::path::PathBuf;

#[test]
fn unsupported_language_returns_structured_error() {
    let file = SourceFile {
        path: PathBuf::from("README.md"),
        language: FileLanguage::Unknown,
        source_text: "# title".to_string(),
    };

    let error = parse_source(file).expect_err("unknown language should fail");
    assert_eq!(error.kind, ErrorKind::UnsupportedLanguage);
    assert_eq!(error.context.operation, Some("parse_source"));
}

#[test]
fn supported_language_returns_empty_parse_result() {
    let file = SourceFile::new("src/lib.rs", "fn main() {}");
    let result = parse_source(file).expect("supported language should parse");

    assert!(result.chunks.is_empty());
    assert!(result.errors.is_empty());
    assert_eq!(result.file.language, FileLanguage::Rust);
}
