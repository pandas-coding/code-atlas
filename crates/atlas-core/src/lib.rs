use std::path::PathBuf;

pub use error::{AtlasError, AtlasResult, ErrorContext, ErrorKind};
pub use model::{
    ChunkKind, ChunkSpan, CodeChunk, FileLanguage, IndexResult, IndexStats, ParseResult, SourceFile,
};

pub mod error;
pub mod model;

pub fn index_path(_path: impl Into<PathBuf>) -> AtlasResult<IndexResult> {
    Ok(IndexResult::default())
}
