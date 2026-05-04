use atlas_core::{ChunkKind, ChunkSpan, CodeChunk, SourceFile};

pub fn build_chunk(
    file: &SourceFile,
    kind: ChunkKind,
    symbol_name: Option<String>,
    span: ChunkSpan,
) -> CodeChunk {
    let source_text = file
        .source_text
        .get(span.start_byte..span.end_byte)
        .unwrap_or_default()
        .to_string();

    CodeChunk {
        file_path: file.path.clone(),
        language: file.language,
        kind,
        symbol_name,
        span,
        source_text,
    }
}
