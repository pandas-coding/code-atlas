use atlas_core::{AtlasResult, ParseResult, SourceFile};

pub trait CodeParser: Send + Sync {
    fn parse_file(&self, source_file: SourceFile) -> AtlasResult<ParseResult>;
}

#[derive(Debug, Default, Clone)]
pub struct ParserService;

impl ParserService {
    pub fn new() -> Self {
        Self
    }
}

impl CodeParser for ParserService {
    fn parse_file(&self, source_file: SourceFile) -> AtlasResult<ParseResult> {
        crate::parse::parse_source(source_file)
    }
}
