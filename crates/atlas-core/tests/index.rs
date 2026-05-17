use atlas_core::{ChunkKind, FileLanguage, IndexOptions, index_path, index_path_with_options};
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
fn index_jsx_fixture() {
    let path = fixtures_dir().join("sample.jsx");
    let result = index_path(&path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.files[0].file.language, FileLanguage::Jsx);
    assert!(result.stats.total_chunks >= 2);

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
}

#[test]
fn index_directory_with_multiple_files() {
    let result = index_path(fixtures_dir(), &parse_source).unwrap();

    assert!(result.stats.total_files >= 3);
    assert_eq!(result.stats.parsed_files, result.stats.total_files);
    assert!(result.errors.is_empty());
    assert!(result.stats.total_chunks >= 13);

    let languages: Vec<_> = result.files.iter().map(|f| f.file.language).collect();
    assert!(languages.contains(&FileLanguage::Rust));
    assert!(languages.contains(&FileLanguage::JavaScript));
    assert!(languages.contains(&FileLanguage::TypeScript));
    assert!(languages.contains(&FileLanguage::Jsx));
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

#[test]
fn index_rust_fixture_all_chunk_kinds() {
    let path = fixtures_dir().join("sample.rs");
    let result = index_path(&path, &parse_source).unwrap();

    let kinds: std::collections::HashSet<_> =
        result.files[0].chunks.iter().map(|c| c.kind).collect();

    assert!(kinds.contains(&ChunkKind::Struct));
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Enum));
    assert!(kinds.contains(&ChunkKind::Trait));
    assert!(kinds.contains(&ChunkKind::Impl));
    assert!(kinds.contains(&ChunkKind::Constant));
    assert!(kinds.contains(&ChunkKind::TypeAlias));
    assert!(kinds.contains(&ChunkKind::Module));
}

#[test]
fn index_javascript_fixture_chunk_names() {
    let path = fixtures_dir().join("sample.js");
    let result = index_path(&path, &parse_source).unwrap();

    let names: Vec<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();

    assert!(names.contains(&"greet"));
    assert!(names.contains(&"Animal"));
    assert!(names.contains(&"API_URL"));
}

#[test]
fn index_typescript_fixture_chunk_names() {
    let path = fixtures_dir().join("sample.ts");
    let result = index_path(&path, &parse_source).unwrap();

    let names: Vec<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();

    assert!(names.contains(&"User"));
    assert!(names.contains(&"ID"));
    assert!(names.contains(&"getUser"));
    assert!(names.contains(&"Service"));
}

#[test]
fn index_directory_all_languages_present() {
    let result = index_path(fixtures_dir(), &parse_source).unwrap();

    let languages: std::collections::HashSet<_> =
        result.files.iter().map(|f| f.file.language).collect();

    assert!(languages.contains(&FileLanguage::Rust));
    assert!(languages.contains(&FileLanguage::JavaScript));
    assert!(languages.contains(&FileLanguage::TypeScript));
}

#[test]
fn index_directory_all_chunks_have_valid_spans() {
    let result = index_path(fixtures_dir(), &parse_source).unwrap();

    for file_result in &result.files {
        for chunk in &file_result.chunks {
            assert!(chunk.span.start_byte < chunk.span.end_byte);
            assert!(chunk.span.start_line <= chunk.span.end_line);
            assert_eq!(chunk.source_text.len(), chunk.span.end_byte - chunk.span.start_byte);
        }
    }
}

#[test]
fn index_empty_temp_directory() {
    let tmp = tempfile::TempDir::new().unwrap();
    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 0);
    assert_eq!(result.stats.parsed_files, 0);
    assert_eq!(result.stats.total_chunks, 0);
    assert_eq!(result.stats.total_errors, 0);
}

#[test]
fn index_single_rust_temp_file() {
    let tmp = tempfile::TempDir::new().unwrap();
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
fn index_single_js_temp_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    let file_path = tmp.path().join("app.js");
    fs::write(&file_path, "function hello() { return 42; }").unwrap();

    let result = index_path(&file_path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.files[0].file.language, FileLanguage::JavaScript);
}

#[test]
fn index_mixed_project_temp_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
    fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
    fs::write(root.join("src/index.js"), "function greet() {}").unwrap();
    fs::write(root.join("src/app.tsx"), "export function App() {}").unwrap();

    fs::create_dir_all(root.join("target/debug")).unwrap();
    fs::write(root.join("target/debug/binary"), "binary").unwrap();

    fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
    fs::write(root.join("node_modules/pkg/index.js"), "module.exports = {};").unwrap();

    let result = index_path(root, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 4);
    assert_eq!(result.stats.parsed_files, 4);

    for file in &result.files {
        let p = file.file.path.to_string_lossy();
        assert!(!p.contains("target"));
        assert!(!p.contains("node_modules"));
    }
}

#[test]
fn index_temp_dir_only_non_source_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("image.png"), "fake").unwrap();
    fs::write(tmp.path().join("data.pdf"), "fake").unwrap();
    fs::write(tmp.path().join("archive.zip"), "fake").unwrap();

    let result = index_path(tmp.path(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 0);
    assert_eq!(result.stats.parsed_files, 0);
}

#[test]
fn index_parallel_consistency_with_many_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    for i in 0..20 {
        let file_name =
            if i % 2 == 0 { format!("src/file_{i:02}.rs") } else { format!("src/file_{i:02}.js") };
        let code = if i % 2 == 0 {
            format!("fn func_{i}() {{}}")
        } else {
            format!("function func_{i}() {{}}")
        };
        fs::write(root.join(&file_name), code).unwrap();
    }

    let r1 = index_path(root, &parse_source).unwrap();
    let r2 = index_path(root, &parse_source).unwrap();

    assert_eq!(r1.stats.total_files, r2.stats.total_files);
    assert_eq!(r1.stats.parsed_files, r2.stats.parsed_files);
    assert_eq!(r1.stats.skipped_files, r2.stats.skipped_files);
    assert_eq!(r1.stats.total_chunks, r2.stats.total_chunks);
    assert_eq!(r1.stats.total_errors, r2.stats.total_errors);
    assert_eq!(r1.stats.files_by_language, r2.stats.files_by_language);
    assert_eq!(r1.stats.chunks_by_language, r2.stats.chunks_by_language);
    assert_eq!(r1.stats.chunks_by_kind, r2.stats.chunks_by_kind);
    assert_eq!(r1.stats.total_source_bytes, r2.stats.total_source_bytes);
}

#[test]
fn index_python_fixture() {
    let path = fixtures_dir().join("sample.py");
    let result = index_path(&path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.stats.total_chunks >= 2);

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
}

#[test]
fn index_php_fixture() {
    let path = fixtures_dir().join("sample.php");
    let result = index_path(&path, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.stats.total_chunks >= 2);

    let kinds: Vec<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
}

#[test]
fn index_gitignore_respected() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(root.join("generated/output.rs"), "fn generated() {}").unwrap();

    fs::create_dir_all(root.join(".git")).unwrap();
    fs::write(root.join(".gitignore"), "generated/\n").unwrap();

    let result = index_path(root, &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 1);
    for file in &result.files {
        let p = file.file.path.to_string_lossy();
        assert!(!p.contains("generated"), "gitignore should skip generated/: {p}");
    }
}

#[test]
fn index_large_file_partial_read() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();

    let mut big_content = String::new();
    for i in 0..2000 {
        big_content.push_str(&format!("fn func_{}() {{}}\n", i));
    }
    fs::write(root.join("big.rs"), &big_content).unwrap();

    let options = IndexOptions {
        large_file_threshold: 100,
        large_file_max_lines: 50,
        ..IndexOptions::default()
    };

    let result = index_path_with_options(root, &parse_source, &options).unwrap();

    assert_eq!(result.stats.total_files, 1);
    assert!(result.files[0].file.is_large);
    let source_len = result.files[0].file.source_text.len();
    assert!(source_len < big_content.len(), "large file should be partially read");
}

#[test]
fn index_chunk_split_threshold() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();

    let mut long_code = String::from("fn big() {\n");
    for i in 0..100 {
        long_code.push_str(&format!("    let x{} = {};\n\n", i, i));
    }
    long_code.push_str("}\n");
    fs::write(root.join("big_fn.rs"), &long_code).unwrap();

    let options = IndexOptions { chunk_split_threshold: 200, ..IndexOptions::default() };

    let result = index_path_with_options(root, &parse_source, &options).unwrap();

    assert!(result.stats.total_chunks > 1, "chunk should be split when exceeding threshold");
}

#[test]
fn index_incremental_skips_unchanged_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let state_path = tmp.path().join(".code-atlas/state.json");

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

    let options = IndexOptions {
        incremental_state_path: Some(state_path.clone()),
        ..IndexOptions::default()
    };

    let result1 = index_path_with_options(root, &parse_source, &options).unwrap();
    assert_eq!(result1.stats.total_files, 1);
    assert_eq!(result1.stats.parsed_files, 1);

    assert!(state_path.exists(), "state file should be created");

    let result2 = index_path_with_options(root, &parse_source, &options).unwrap();
    assert_eq!(result2.stats.total_files, 0, "unchanged files should be skipped");
    assert_eq!(result2.stats.parsed_files, 0);
}

#[test]
fn index_incremental_reindexes_changed_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    let root = tmp.path();
    let state_path = tmp.path().join(".code-atlas/state.json");

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

    let options = IndexOptions {
        incremental_state_path: Some(state_path.clone()),
        ..IndexOptions::default()
    };

    let result1 = index_path_with_options(root, &parse_source, &options).unwrap();
    assert_eq!(result1.stats.parsed_files, 1);

    std::thread::sleep(std::time::Duration::from_millis(100));
    fs::write(root.join("src/main.rs"), "fn updated() {}").unwrap();

    let result2 = index_path_with_options(root, &parse_source, &options).unwrap();
    assert_eq!(result2.stats.parsed_files, 1, "changed files should be re-indexed");
    assert_eq!(result2.stats.total_chunks, 1);
}
