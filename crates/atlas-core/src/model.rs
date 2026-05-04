use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileLanguage {
    Rust,
    JavaScript,
    TypeScript,
    Jsx,
    Tsx,
    Unknown,
}

impl FileLanguage {
    pub fn from_extension(extension: &str) -> Self {
        match extension {
            "rs" => Self::Rust,
            "js" => Self::JavaScript,
            "jsx" => Self::Jsx,
            "ts" => Self::TypeScript,
            "tsx" => Self::Tsx,
            _ => Self::Unknown,
        }
    }

    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Unknown)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Jsx => "jsx",
            Self::Tsx => "tsx",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FileLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub path: PathBuf,
    pub language: FileLanguage,
    pub source_text: String,
}

impl SourceFile {
    pub fn new(path: impl Into<PathBuf>, source_text: impl Into<String>) -> Self {
        let path = path.into();
        let language = FileLanguage::from_path(&path);

        Self {
            path,
            language,
            source_text: source_text.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}

impl ChunkSpan {
    pub fn new(start_byte: usize, end_byte: usize, start_line: usize, end_line: usize) -> Self {
        Self {
            start_byte,
            end_byte,
            start_line,
            end_line,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Class,
    Interface,
    TypeAlias,
    Module,
    Constant,
    Method,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeChunk {
    pub file_path: PathBuf,
    pub language: FileLanguage,
    pub kind: ChunkKind,
    pub symbol_name: Option<String>,
    pub span: ChunkSpan,
    pub source_text: String,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub file: SourceFile,
    pub chunks: Vec<CodeChunk>,
    pub errors: Vec<crate::error::AtlasError>,
}

impl ParseResult {
    pub fn success(file: SourceFile, chunks: Vec<CodeChunk>) -> Self {
        Self {
            file,
            chunks,
            errors: Vec::new(),
        }
    }

    pub fn with_errors(mut self, errors: Vec<crate::error::AtlasError>) -> Self {
        self.errors = errors;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub total_files: usize,
    pub parsed_files: usize,
    pub skipped_files: usize,
    pub total_chunks: usize,
    pub total_errors: usize,
}

#[derive(Debug, Clone, Default)]
pub struct IndexResult {
    pub files: Vec<ParseResult>,
    pub errors: Vec<crate::error::AtlasError>,
    pub stats: IndexStats,
}

impl IndexResult {
    pub fn from_parse_results(
        files: Vec<ParseResult>,
        errors: Vec<crate::error::AtlasError>,
    ) -> Self {
        let parsed_files = files.len();
        let total_chunks = files.iter().map(|file| file.chunks.len()).sum();
        let file_errors: usize = files.iter().map(|file| file.errors.len()).sum();
        let total_errors = errors.len() + file_errors;
        let total_files = parsed_files + errors.len();

        Self {
            stats: IndexStats {
                total_files,
                parsed_files,
                skipped_files: total_files.saturating_sub(parsed_files),
                total_chunks,
                total_errors,
            },
            files,
            errors,
        }
    }
}
