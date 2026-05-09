use atlas_core::{index_path, ChunkKind, FileLanguage};
use atlas_parser::parse_source;
use std::fs;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn index_rust_fixture() {
    let path = fixtures_dir().join("sample.rs");
    let result = index_path(&path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.stats.total_chunks >= 7);

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Struct));
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Enum));
    assert!(kinds.contains(&ChunkKind::Trait));
    assert!(kinds.contains(&ChunkKind::Impl));
    assert!(kinds.contains(&ChunkKind::Constant));
    assert!(kinds.contains(&ChunkKind::TypeAlias));
}

#[test]
fn index_javascript_fixture() {
    let path = fixtures_dir().join("sample.js");
    let result = index_path(&path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert!(result.stats.total_chunks >= 3);

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
    assert!(kinds.contains(&ChunkKind::Constant));
}

#[test]
fn index_typescript_fixture() {
    let path = fixtures_dir().join("sample.ts");
    let result = index_path(&path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert!(result.stats.total_chunks >= 3);

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Interface));
    assert!(kinds.contains(&ChunkKind::TypeAlias));
    assert!(kinds.contains(&ChunkKind::Function));
}

#[test]
fn index_directory_with_multiple_files() {
    let result = index_path(fixtures_dir(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 3);
    assert_eq!(result.stats.parsed_files, 3);
    assert!(result.errors.is_empty());
    assert!(result.stats.total_chunks >= 13);

    let languages: Vec<_> = result
        .files
        .iter()
        .map(|f| f.file.language)
        .collect();
    assert!(languages.contains(&FileLanguage::Rust));
    assert!(languages.contains(&FileLanguage::JavaScript));
    assert!(languages.contains(&FileLanguage::TypeScript));
}

#[test]
fn index_nonexistent_path_returns_error() {
    let result = index_path("/nonexistent/path", &parse_source);
    assert!(result.is_err());
}

#[test]
fn chunk_symbol_names_are_extracted() {
    let path = fixtures_dir().join("sample.rs");
    let result = index_path(&path, &parse_source).unwrap();

    let names: Vec<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();

    assert!(names.contains(&"Point"));
    assert!(names.contains(&"new_point"));
    assert!(names.contains(&"Shape"));
    assert!(names.contains(&"Drawable"));
}

#[test]
fn chunk_spans_are_valid() {
    let path = fixtures_dir().join("sample.rs");
    let result = index_path(&path, &parse_source).unwrap();

    for chunk in &result.files[0].chunks {
        assert!(chunk.span.start_byte < chunk.span.end_byte);
        assert!(chunk.span.start_line <= chunk.span.end_line);
        assert_eq!(chunk.source_text.len(), chunk.span.end_byte - chunk.span.start_byte);
    }
}

#[test]
fn parallel_index_produces_consistent_results() {
    let r1 = index_path(fixtures_dir(), &parse_source).unwrap();
    let r2 = index_path(fixtures_dir(), &parse_source).unwrap();

    assert_eq!(r1.stats.total_files, r2.stats.total_files);
    assert_eq!(r1.stats.total_chunks, r2.stats.total_chunks);
}

#[test]
fn index_with_temp_project_containing_ignored_dirs() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

    fs::create_dir_all(root.join("target")).unwrap();
    fs::write(root.join("target/build.rs"), "fn build() {}").unwrap();

    fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
    fs::write(root.join("node_modules/pkg/index.js"), "function x() {}").unwrap();

    let result = index_path(root, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);

    for file in &result.files {
        let p = file.file.path.to_string_lossy();
        assert!(!p.contains("target"));
        assert!(!p.contains("node_modules"));
    }
}
