use atlas_core::{AtlasError, AtlasResult, ErrorContext, FileLanguage, ParseResult, SourceFile};

use crate::ParserService;

pub fn parse_source(source_file: SourceFile) -> AtlasResult<ParseResult> {
    if !source_file.language.is_supported() {
        return Err(
            AtlasError::unsupported_language("source file language is not supported yet")
                .with_context(
                    ErrorContext::default()
                        .with_operation("parse_source")
                        .with_path(source_file.path.clone())
                        .with_language(source_file.language),
                ),
        );
    }

    Ok(ParseResult::success(source_file, Vec::new()))
}

pub fn parser_for_language(language: FileLanguage) -> AtlasResult<ParserService> {
    if language.is_supported() {
        Ok(ParserService::new())
    } else {
        Err(
            AtlasError::unsupported_language("parser is not available for this language")
                .with_context(
                    ErrorContext::default()
                        .with_operation("parser_for_language")
                        .with_language(language),
                ),
        )
    }
}
