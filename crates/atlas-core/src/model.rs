use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Supported programming languages for source file parsing.
///
/// Each variant corresponds to a language that atlas can parse and extract
/// code chunks from. The `Unknown` variant represents files with unsupported
/// or unrecognized extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileLanguage {
    /// Rust source files (`.rs`).
    Rust,
    /// JavaScript source files (`.js`).
    JavaScript,
    /// TypeScript source files (`.ts`).
    TypeScript,
    /// JSX files (`.jsx`).
    Jsx,
    /// TSX files (`.tsx`).
    Tsx,
    /// Python source files (`.py`).
    Python,
    /// PHP source files (`.php`).
    Php,
    /// Unsupported or unrecognized file extension.
    Unknown,
}

impl FileLanguage {
    /// Maps a file extension string to the corresponding [`FileLanguage`].
    ///
    /// Returns `Unknown` for unrecognized extensions.
    pub fn from_extension(extension: &str) -> Self {
        match extension {
            "rs" => Self::Rust,
            "js" => Self::JavaScript,
            "jsx" => Self::Jsx,
            "ts" => Self::TypeScript,
            "tsx" => Self::Tsx,
            "py" => Self::Python,
            "php" => Self::Php,
            _ => Self::Unknown,
        }
    }

    /// Returns a slice of all supported language variants (excluding `Unknown`).
    pub fn all_supported() -> &'static [FileLanguage] {
        &[
            Self::Rust,
            Self::JavaScript,
            Self::TypeScript,
            Self::Jsx,
            Self::Tsx,
            Self::Python,
            Self::Php,
        ]
    }

    /// Detects the language from a file path based on its extension.
    ///
    /// Returns `Unknown` if the path has no extension or an unsupported extension.
    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Returns `true` if this language variant is supported for parsing.
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Unknown)
    }

    /// Returns the lowercase string representation of the language.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Jsx => "jsx",
            Self::Tsx => "tsx",
            Self::Python => "python",
            Self::Php => "php",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FileLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A source file with its detected language and text content.
///
/// This is the primary input unit for the parsing pipeline. The language
/// is automatically detected from the file path extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    /// Path to the source file.
    pub path: PathBuf,
    /// Detected programming language.
    pub language: FileLanguage,
    /// Full text content of the source file.
    pub source_text: String,
    /// Whether this file exceeds the large-file threshold.
    pub is_large: bool,
}

impl SourceFile {
    /// Creates a new `SourceFile` with language auto-detected from the path.
    pub fn new(path: impl Into<PathBuf>, source_text: impl Into<String>) -> Self {
        let path = path.into();
        let language = FileLanguage::from_path(&path);

        Self { path, language, source_text: source_text.into(), is_large: false }
    }
}

/// Byte and line range of a code chunk within its source file.
///
/// Line numbers are zero-indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    /// Starting byte offset (inclusive).
    pub start_byte: usize,
    /// Ending byte offset (exclusive).
    pub end_byte: usize,
    /// Starting line number (zero-indexed).
    pub start_line: usize,
    /// Ending line number (zero-indexed).
    pub end_line: usize,
}

impl ChunkSpan {
    /// Creates a new span with the given byte and line ranges.
    pub fn new(start_byte: usize, end_byte: usize, start_line: usize, end_line: usize) -> Self {
        Self { start_byte, end_byte, start_line, end_line }
    }
}

/// Classification of code chunk types extracted from AST nodes.
///
/// Each variant represents a distinct syntactic construct such as a function,
/// struct, class, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    /// Function or method definition.
    Function,
    /// Struct definition (Rust).
    Struct,
    /// Enum definition.
    Enum,
    /// Trait definition (Rust).
    Trait,
    /// Impl block (Rust).
    Impl,
    /// Class definition (JS/TS).
    Class,
    /// Interface definition (TS).
    Interface,
    /// Type alias definition.
    TypeAlias,
    /// Module declaration.
    Module,
    /// Constant definition.
    Constant,
    /// Method definition within a class or impl block.
    Method,
    /// Unrecognized or unsupported chunk type.
    Unknown,
}

impl fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Class => "class",
            Self::Interface => "interface",
            Self::TypeAlias => "type_alias",
            Self::Module => "module",
            Self::Constant => "constant",
            Self::Method => "method",
            Self::Unknown => "unknown",
        };

        write!(f, "{value}")
    }
}

/// A code chunk extracted from a source file.
///
/// Represents a single meaningful syntactic unit (function, struct, class, etc.)
/// identified during parsing. Each chunk has a unique identifier, location span,
/// and optional symbol name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeChunk {
    /// Unique identifier for this chunk (generated from path, kind, name, line).
    pub id: String,
    /// Path to the source file containing this chunk.
    pub file_path: PathBuf,
    /// Programming language of the source file.
    pub language: FileLanguage,
    /// Type of code construct this chunk represents.
    pub kind: ChunkKind,
    /// Extracted symbol name (e.g., function name), if available.
    pub symbol_name: Option<String>,
    /// Byte and line range of this chunk within the source file.
    pub span: ChunkSpan,
    /// Source code text of this chunk.
    pub source_text: String,
}

impl CodeChunk {
    /// Generates a deterministic unique ID for a chunk.
    ///
    /// Format: `<file_path>:<kind>:<symbol_name>:<start_line>`
    pub fn generate_id(
        file_path: &Path,
        kind: ChunkKind,
        symbol_name: Option<&str>,
        start_line: usize,
    ) -> String {
        let file_str = file_path.to_string_lossy();
        let symbol = symbol_name.unwrap_or("_");
        format!("{}:{}:{}:{}", file_str, kind, symbol, start_line)
    }
}

/// Result of parsing a single source file.
///
/// Contains the original file, extracted chunks, and any parse errors
/// (syntax errors are recoverable and don't prevent chunk extraction).
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// The original source file that was parsed.
    pub file: SourceFile,
    /// Code chunks extracted from the file.
    pub chunks: Vec<CodeChunk>,
    /// Parse errors or warnings encountered during parsing.
    pub errors: Vec<crate::error::AtlasError>,
}

impl ParseResult {
    /// Creates a successful parse result with no errors.
    pub fn success(file: SourceFile, chunks: Vec<CodeChunk>) -> Self {
        Self { file, chunks, errors: Vec::new() }
    }

    /// Attaches parse errors/warnings to this result.
    pub fn with_errors(mut self, errors: Vec<crate::error::AtlasError>) -> Self {
        self.errors = errors;
        self
    }
}

/// Aggregated statistics from an indexing operation.
///
/// Provides counts and breakdowns by language and chunk kind,
/// along with timing information.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexStats {
    /// Total number of files encountered (parsed + skipped/errored).
    pub total_files: usize,
    /// Number of files successfully parsed.
    pub parsed_files: usize,
    /// Number of files skipped (unsupported language, binary, etc.).
    pub skipped_files: usize,
    /// Total number of code chunks extracted.
    pub total_chunks: usize,
    /// Total number of errors encountered.
    pub total_errors: usize,
    /// File counts grouped by language.
    pub files_by_language: HashMap<FileLanguage, usize>,
    /// Chunk counts grouped by language.
    pub chunks_by_language: HashMap<FileLanguage, usize>,
    /// Chunk counts grouped by chunk kind.
    pub chunks_by_kind: HashMap<ChunkKind, usize>,
    /// Total bytes of source code processed.
    pub total_source_bytes: usize,
    /// Total elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Number of chunks successfully embedded (0 if embedding disabled).
    pub embedded_chunks: usize,
    /// Number of chunks that failed during embedding (0 if embedding disabled).
    pub embedding_errors: usize,
    /// Dimension of the embedding vectors (0 if embedding disabled).
    pub embedding_dimension: usize,
}

/// Complete result of an indexing operation across multiple files.
///
/// Contains all per-file parse results, top-level errors, and aggregated
/// statistics. Constructed via [`IndexResult::from_parse_results`].
#[derive(Debug, Clone, Default)]
pub struct IndexResult {
    /// Individual file parse results.
    pub files: Vec<ParseResult>,
    /// Top-level errors (e.g., files that couldn't be read).
    pub errors: Vec<crate::error::AtlasError>,
    /// Aggregated statistics across all files.
    pub stats: IndexStats,
}

impl IndexResult {
    /// Aggregates per-file parse results and top-level errors into an [`IndexResult`].
    ///
    /// Computes all statistics (file counts, chunk counts, language/kind breakdowns).
    /// Elapsed time must be set separately via [`set_elapsed_ms`](Self::set_elapsed_ms).
    pub fn from_parse_results(
        files: Vec<ParseResult>,
        errors: Vec<crate::error::AtlasError>,
    ) -> Self {
        let parsed_files = files.len();
        let total_chunks = files.iter().map(|file| file.chunks.len()).sum();
        let file_errors: usize = files.iter().map(|file| file.errors.len()).sum();
        let total_errors = errors.len() + file_errors;
        let total_files = parsed_files + errors.len();

        let mut files_by_language: HashMap<FileLanguage, usize> = HashMap::new();
        let mut chunks_by_language: HashMap<FileLanguage, usize> = HashMap::new();
        let mut chunks_by_kind: HashMap<ChunkKind, usize> = HashMap::new();
        let mut total_source_bytes = 0;

        for file in &files {
            let lang = file.file.language;
            *files_by_language.entry(lang).or_insert(0) += 1;
            total_source_bytes += file.file.source_text.len();

            for chunk in &file.chunks {
                *chunks_by_language.entry(chunk.language).or_insert(0) += 1;
                *chunks_by_kind.entry(chunk.kind).or_insert(0) += 1;
            }
        }

        Self {
            stats: IndexStats {
                total_files,
                parsed_files,
                skipped_files: total_files.saturating_sub(parsed_files),
                total_chunks,
                total_errors,
                files_by_language,
                chunks_by_language,
                chunks_by_kind,
                total_source_bytes,
                elapsed_ms: 0,
                embedded_chunks: 0,
                embedding_errors: 0,
                embedding_dimension: 0,
            },
            files,
            errors,
        }
    }

    /// Sets the elapsed time for the indexing operation in milliseconds.
    pub fn set_elapsed_ms(&mut self, ms: u64) {
        self.stats.elapsed_ms = ms;
    }
}

/// Embedding options for the indexing pipeline.
///
/// When provided, the indexer will generate vector embeddings for all
/// code chunks and persist the vector store to disk.
#[derive(Debug, Clone)]
pub struct EmbeddingOptions {
    /// Configuration for the embedding model (model path, dimension, etc.).
    pub config: atlas_vdb::EmbeddingConfig,
    /// Path to save the vector store file after indexing.
    pub vector_store_path: PathBuf,
    /// Optional batch size for embedding inference (default: 32).
    pub batch_size: usize,
}

/// Configuration options for the indexing pipeline.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Path to the incremental index state file.
    ///
    /// If set, the indexer will load the previous state and skip
    /// unchanged files, then save the updated state after indexing.
    pub incremental_state_path: Option<PathBuf>,
    /// File size threshold in bytes above which a file is treated as "large".
    ///
    /// Large files may be processed differently (e.g., partial reads,
    /// limited chunk extraction). Default: 1 MB.
    pub large_file_threshold: usize,
    /// Maximum number of lines to read from a large file.
    ///
    /// When a file exceeds the large-file threshold, only the first
    /// `large_file_max_lines` lines are read. Default: 500.
    pub large_file_max_lines: usize,
    /// Maximum number of characters a chunk can have before being split.
    ///
    /// Chunks exceeding this threshold are subdivided into smaller
    /// pieces. Default: 3000.
    pub chunk_split_threshold: usize,
    /// Embedding options. When `None`, no embedding is performed (M1 behavior).
    pub embedding: Option<EmbeddingOptions>,
}

impl IndexOptions {
    /// Creates `IndexOptions` with default values.
    pub fn new() -> Self {
        Self {
            incremental_state_path: None,
            large_file_threshold: 1024 * 1024,
            large_file_max_lines: 500,
            chunk_split_threshold: 3000,
            embedding: None,
        }
    }
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Recorded state for a single file in the incremental index.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileState {
    /// Last-modified timestamp (seconds since epoch).
    pub mtime_secs: u64,
    /// Last-modified timestamp sub-second nanoseconds.
    pub mtime_nanos: u32,
    /// File size in bytes at the time of indexing.
    pub size: u64,
    /// Number of chunks extracted from this file.
    pub chunk_count: usize,
}

/// Persistent state for incremental indexing.
///
/// Stored as JSON so it can be inspected and debugged easily.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexState {
    /// State format version for forward compatibility.
    pub version: u32,
    /// Per-file state keyed by normalized path string.
    pub file_states: HashMap<String, FileState>,
}

impl IndexState {
    /// Current state format version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Creates an empty state with the current version.
    pub fn new() -> Self {
        Self { version: Self::CURRENT_VERSION, file_states: HashMap::new() }
    }

    /// Loads state from a JSON file.
    pub fn load(path: &Path) -> crate::AtlasResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            crate::AtlasError::io(format!("Failed to read index state: {}", e))
                .with_context(
                    crate::ErrorContext::default()
                        .with_operation("load_state")
                        .with_path(path),
                )
                .with_source(e.to_string())
        })?;
        let state: Self = serde_json::from_str(&text).map_err(|e| {
            crate::AtlasError::internal(format!("Failed to parse index state: {}", e))
                .with_context(
                    crate::ErrorContext::default()
                        .with_operation("load_state")
                        .with_path(path),
                )
                .with_source(e.to_string())
        })?;
        Ok(state)
    }

    /// Saves state to a JSON file.
    pub fn save(&self, path: &Path) -> crate::AtlasResult<()> {
        let text = serde_json::to_string_pretty(self).map_err(|e| {
            crate::AtlasError::internal(format!("Failed to serialize index state: {}", e))
                .with_context(
                    crate::ErrorContext::default()
                        .with_operation("save_state")
                        .with_path(path),
                )
                .with_source(e.to_string())
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::AtlasError::io(format!("Failed to create state directory: {}", e))
                    .with_context(
                        crate::ErrorContext::default()
                            .with_operation("save_state")
                            .with_path(parent),
                    )
                    .with_source(e.to_string())
            })?;
        }
        std::fs::write(path, text).map_err(|e| {
            crate::AtlasError::io(format!("Failed to write index state: {}", e))
                .with_context(
                    crate::ErrorContext::default()
                        .with_operation("save_state")
                        .with_path(path),
                )
                .with_source(e.to_string())
        })?;
        Ok(())
    }

    /// Checks whether a file has changed since the last index.
    pub fn has_file_changed(&self, path: &Path, mtime: std::time::SystemTime, size: u64) -> bool {
        let key = path.to_string_lossy().to_string();
        match self.file_states.get(&key) {
            Some(prev) => {
                let dur = mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = dur.as_secs();
                let nanos = dur.subsec_nanos();
                prev.mtime_secs != secs || prev.mtime_nanos != nanos || prev.size != size
            }
            None => true,
        }
    }

    /// Records the state of a file after successful indexing.
    pub fn record_file(
        &mut self,
        path: &Path,
        mtime: std::time::SystemTime,
        size: u64,
        chunk_count: usize,
    ) {
        let dur = mtime
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let key = path.to_string_lossy().to_string();
        self.file_states.insert(
            key,
            FileState {
                mtime_secs: dur.as_secs(),
                mtime_nanos: dur.subsec_nanos(),
                size,
                chunk_count,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_extension_rust() {
        assert_eq!(FileLanguage::from_extension("rs"), FileLanguage::Rust);
    }

    #[test]
    fn from_extension_javascript() {
        assert_eq!(FileLanguage::from_extension("js"), FileLanguage::JavaScript);
    }

    #[test]
    fn from_extension_jsx() {
        assert_eq!(FileLanguage::from_extension("jsx"), FileLanguage::Jsx);
    }

    #[test]
    fn from_extension_typescript() {
        assert_eq!(FileLanguage::from_extension("ts"), FileLanguage::TypeScript);
    }

    #[test]
    fn from_extension_tsx() {
        assert_eq!(FileLanguage::from_extension("tsx"), FileLanguage::Tsx);
    }

    #[test]
    fn from_extension_unknown() {
        assert_eq!(FileLanguage::from_extension("java"), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_extension("go"), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_extension(""), FileLanguage::Unknown);
    }

    #[test]
    fn from_path_simple_filenames() {
        assert_eq!(FileLanguage::from_path(Path::new("main.rs")), FileLanguage::Rust);
        assert_eq!(FileLanguage::from_path(Path::new("index.js")), FileLanguage::JavaScript);
        assert_eq!(FileLanguage::from_path(Path::new("App.tsx")), FileLanguage::Tsx);
        assert_eq!(FileLanguage::from_path(Path::new("Component.jsx")), FileLanguage::Jsx);
        assert_eq!(FileLanguage::from_path(Path::new("types.ts")), FileLanguage::TypeScript);
    }

    #[test]
    fn from_path_with_directory() {
        assert_eq!(FileLanguage::from_path(Path::new("src/lib.rs")), FileLanguage::Rust);
        assert_eq!(
            FileLanguage::from_path(Path::new("project/src/index.js")),
            FileLanguage::JavaScript
        );
    }

    #[test]
    fn from_path_no_extension() {
        assert_eq!(FileLanguage::from_path(Path::new("Makefile")), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_path(Path::new("README")), FileLanguage::Unknown);
    }

    #[test]
    fn from_path_unsupported_extension() {
        assert_eq!(FileLanguage::from_path(Path::new("style.css")), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_path(Path::new("data.json")), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_path(Path::new("doc.md")), FileLanguage::Unknown);
    }

    #[test]
    fn is_supported() {
        assert!(FileLanguage::Rust.is_supported());
        assert!(FileLanguage::JavaScript.is_supported());
        assert!(FileLanguage::TypeScript.is_supported());
        assert!(FileLanguage::Jsx.is_supported());
        assert!(FileLanguage::Tsx.is_supported());
        assert!(FileLanguage::Python.is_supported());
        assert!(FileLanguage::Php.is_supported());
        assert!(!FileLanguage::Unknown.is_supported());
    }

    #[test]
    fn as_str_roundtrip() {
        for lang in FileLanguage::all_supported() {
            assert_eq!(lang.as_str(), lang.to_string());
        }
    }

    #[test]
    fn display_format() {
        assert_eq!(FileLanguage::Rust.to_string(), "rust");
        assert_eq!(FileLanguage::JavaScript.to_string(), "javascript");
        assert_eq!(FileLanguage::TypeScript.to_string(), "typescript");
        assert_eq!(FileLanguage::Jsx.to_string(), "jsx");
        assert_eq!(FileLanguage::Tsx.to_string(), "tsx");
        assert_eq!(FileLanguage::Unknown.to_string(), "unknown");
    }

    #[test]
    fn source_file_detects_language_from_path() {
        let sf = SourceFile::new("lib.rs", "fn main() {}");
        assert_eq!(sf.language, FileLanguage::Rust);

        let sf = SourceFile::new("app.tsx", "export default function App() {}");
        assert_eq!(sf.language, FileLanguage::Tsx);
    }

    #[test]
    fn source_file_unknown_language() {
        let sf = SourceFile::new("style.css", "body {}");
        assert_eq!(sf.language, FileLanguage::Unknown);
    }

    #[test]
    fn chunk_span_fields() {
        let span = ChunkSpan::new(10, 50, 2, 5);
        assert_eq!(span.start_byte, 10);
        assert_eq!(span.end_byte, 50);
        assert_eq!(span.start_line, 2);
        assert_eq!(span.end_line, 5);
    }

    #[test]
    fn chunk_kind_display() {
        assert_eq!(ChunkKind::Function.to_string(), "function");
        assert_eq!(ChunkKind::Struct.to_string(), "struct");
        assert_eq!(ChunkKind::Enum.to_string(), "enum");
        assert_eq!(ChunkKind::Trait.to_string(), "trait");
        assert_eq!(ChunkKind::Impl.to_string(), "impl");
        assert_eq!(ChunkKind::Class.to_string(), "class");
        assert_eq!(ChunkKind::Interface.to_string(), "interface");
        assert_eq!(ChunkKind::TypeAlias.to_string(), "type_alias");
        assert_eq!(ChunkKind::Module.to_string(), "module");
        assert_eq!(ChunkKind::Constant.to_string(), "constant");
        assert_eq!(ChunkKind::Method.to_string(), "method");
        assert_eq!(ChunkKind::Unknown.to_string(), "unknown");
    }

    #[test]
    fn index_stats_default() {
        let stats = IndexStats::default();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.parsed_files, 0);
        assert_eq!(stats.skipped_files, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn index_result_from_parse_results_counts() {
        let file1 = SourceFile::new("a.rs", "fn a() {}");
        let chunk1 = CodeChunk {
            id: CodeChunk::generate_id(Path::new("a.rs"), ChunkKind::Function, Some("a"), 0),
            file_path: PathBuf::from("a.rs"),
            language: FileLanguage::Rust,
            kind: ChunkKind::Function,
            symbol_name: Some("a".into()),
            span: ChunkSpan::new(0, 10, 0, 0),
            source_text: "fn a() {}".into(),
        };
        let pr1 = ParseResult::success(file1, vec![chunk1]);

        let file2 = SourceFile::new("b.rs", "fn b() {}");
        let chunk2 = CodeChunk {
            id: CodeChunk::generate_id(Path::new("b.rs"), ChunkKind::Function, Some("b"), 0),
            file_path: PathBuf::from("b.rs"),
            language: FileLanguage::Rust,
            kind: ChunkKind::Function,
            symbol_name: Some("b".into()),
            span: ChunkSpan::new(0, 10, 0, 0),
            source_text: "fn b() {}".into(),
        };
        let pr2 = ParseResult::success(file2, vec![chunk2]);

        let result = IndexResult::from_parse_results(vec![pr1, pr2], vec![]);
        assert_eq!(result.stats.total_files, 2);
        assert_eq!(result.stats.parsed_files, 2);
        assert_eq!(result.stats.total_chunks, 2);
        assert_eq!(result.stats.total_errors, 0);
        assert_eq!(result.stats.files_by_language.get(&FileLanguage::Rust), Some(&2));
        assert_eq!(result.stats.chunks_by_kind.get(&ChunkKind::Function), Some(&2));
    }
}
