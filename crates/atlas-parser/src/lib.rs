pub mod chunk;
pub mod parse;
pub mod parser;

// Re-export for convenience
pub use chunk::build_chunk;
pub use parse::{parse_source, parser_for_language};
pub use parser::{CodeParser, ParserService};

#[cfg(test)]
mod tests;
