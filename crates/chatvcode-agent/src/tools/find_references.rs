use std::fs;
use std::io::{BufRead, BufReader};

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use regex::Regex;
use serde_json::Value;
use walkdir::WalkDir;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

/// `find_references` 工具：在代码库中查找指定符号的引用位置。
///
/// 通过正则匹配（带单词边界）扫描项目文件，返回符号出现的位置。
/// 与 `search_symbol`（基于索引）互补，可在未索引的项目中使用，
/// 且能捕获到用法（调用、引用）而非仅定义。
pub struct FindReferencesTool;

impl BuiltinTool for FindReferencesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("find_references")
            .description(
                "Find references to a symbol (function, variable, type, etc.) across the \
                 codebase using word-boundary matching. Returns file paths, line numbers, \
                 and the matching line content. Useful for locating usages and call sites.",
            )
            .parameter(
                ToolParameter::string("symbol")
                    .description("Symbol name to find references for (exact word match)")
                    .required(true),
            )
            .parameter(
                ToolParameter::string("path")
                    .description(
                        "Directory or file to search in (relative to project root, default: \".\")",
                    ),
            )
            .parameter(
                ToolParameter::string("file_pattern")
                    .description("Glob pattern to filter files (e.g., \"*.rs\", \"*.ts\")"),
            )
            .parameter(
                ToolParameter::integer("max_results")
                    .description("Maximum number of references to return (default: 100)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let symbol = call.get_string("symbol").unwrap();
        let search_path = call.get_string("path").unwrap_or(".");
        let file_pattern = call.get_string("file_pattern");
        let max_results = call.get_i64("max_results").unwrap_or(100).max(1) as usize;

        // 构建带单词边界的正则：\b<symbol>\b
        // 对符号中的正则元字符进行转义
        let escaped = regex::escape(symbol);
        let pattern = format!(r"\b{}\b", escaped);
        let regex = Regex::new(&pattern).map_err(|e| AgentError::ToolError {
            tool_name: "find_references".into(),
            message: format!("Invalid symbol pattern '{}': {}", symbol, e),
        })?;

        let target = if search_path == "." {
            ctx.project_path.clone()
        } else {
            ctx.project_path.join(search_path)
        };

        if !target.exists() {
            return Ok(ToolResult::error(format!("Path does not exist: {}", search_path)));
        }

        let canonical_project = ctx
            .project_path
            .canonicalize()
            .unwrap_or_else(|_| ctx.project_path.clone());

        let mut results: Vec<Value> = Vec::new();

        if target.is_file() {
            search_file_for_references(
                &target,
                &regex,
                symbol,
                &canonical_project,
                &mut results,
                max_results,
            )?;
        } else {
            for entry in WalkDir::new(&target).follow_links(true) {
                if results.len() >= max_results {
                    break;
                }
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.file_type().is_file() {
                    continue;
                }

                let rel_path = entry
                    .path()
                    .strip_prefix(&canonical_project)
                    .unwrap_or(entry.path())
                    .to_string_lossy()
                    .replace('\\', "/");

                if let Some(ref pat) = file_pattern {
                    let file_name = entry.file_name().to_string_lossy();
                    if !simple_glob_match(&file_name, pat) && !simple_glob_match(&rel_path, pat) {
                        continue;
                    }
                }

                search_file_for_references(
                    entry.path(),
                    &regex,
                    symbol,
                    &canonical_project,
                    &mut results,
                    max_results,
                )?;
            }
        }

        let result = serde_json::json!({
            "symbol": symbol,
            "path": search_path,
            "reference_count": results.len(),
            "truncated": results.len() >= max_results,
            "references": results,
        });

        Ok(ToolResult::success(result))
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            return format!("Error: {}", result.value);
        }

        let count = result.value["reference_count"].as_u64().unwrap_or(0);
        if count == 0 {
            return "No references found".to_string();
        }

        let refs = match result.value["references"].as_array() {
            Some(arr) => arr,
            None => return format!("Found {} references", count),
        };

        let preview: Vec<String> = refs
            .iter()
            .take(5)
            .map(|r| {
                let file = r["file"].as_str().unwrap_or("?");
                let line = r["line"].as_u64().unwrap_or(0);
                format!("{}:{}", file, line)
            })
            .collect();

        format!("Found {} references:\n{}", count, preview.join("\n"))
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

fn search_file_for_references(
    path: &std::path::Path,
    regex: &Regex,
    symbol: &str,
    project_root: &std::path::Path,
    results: &mut Vec<Value>,
    max_results: usize,
) -> Result<(), AgentError> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };

    let reader = BufReader::new(file);
    let rel_path = path
        .strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");

    for (line_num, line_result) in reader.lines().enumerate() {
        if results.len() >= max_results {
            break;
        }
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };
        if regex.is_match(&line) {
            results.push(serde_json::json!({
                "file": rel_path,
                "line": line_num + 1,
                "symbol": symbol,
                "text": line.trim(),
            }));
        }
    }

    Ok(())
}

fn simple_glob_match(name: &str, pattern: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name == pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreTrait, CodeSearchService};
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    struct MockSearch;
    impl CodeSearchService for MockSearch {
        fn search(&self, _: &str, _: usize) -> Result<Vec<SearchResult>, AgentError> {
            Ok(vec![])
        }
    }

    struct MockChunkStore;
    impl ChunkMetadataStoreTrait for MockChunkStore {
        fn get_chunks_by_symbol(&self, _: &str, _: Option<&str>) -> Vec<ChunkMetadata> {
            vec![]
        }
        fn get_chunk_by_id(&self, _: &str) -> Option<ChunkMetadata> {
            None
        }
    }

    fn make_ctx(project_path: PathBuf) -> ToolContext {
        ToolContext {
            project_path,
            timeout: Duration::from_secs(30),
            token_budget: 4096,
            services: Arc::new(AgentServices {
                search: Box::new(MockSearch),
                parser: Box::new(
                    |_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<
                        chatvcode_core::model::ParseResult,
                    > { unimplemented!() },
                ),
                chunk_store: Box::new(MockChunkStore),
            }),
        }
    }

    #[test]
    fn test_find_references_basic() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("a.rs"),
            "fn foo() {}\nfn caller() { foo(); }\nlet x = foo();\n",
        )
        .unwrap();
        fs::write(dir.path().join("b.rs"), "foo_bar();\nfoo!();\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = FindReferencesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("foo".into()));
        let call = ToolCall { name: "find_references".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        // "foo" 出现在 a.rs 的 3 行（定义、调用、赋值）；b.rs 中 foo_bar 不应匹配（单词边界）
        let count = result.value["reference_count"].as_u64().unwrap();
        assert!(count >= 3, "expected at least 3 references, got {}", count);
    }

    #[test]
    fn test_find_references_word_boundary() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "foo\nfoobar\nfoo_bar\nfoo!\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = FindReferencesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("foo".into()));
        let call = ToolCall { name: "find_references".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        // foo (line 1), foo! (line 4) — foobar 和 foo_bar 不应匹配
        let count = result.value["reference_count"].as_u64().unwrap();
        assert_eq!(count, 2, "expected 2 references with word boundary, got {}", count);
    }

    #[test]
    fn test_find_references_with_file_pattern() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "my_func()\n").unwrap();
        fs::write(dir.path().join("b.txt"), "my_func\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = FindReferencesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("my_func".into()));
        args.insert("file_pattern".to_string(), Value::String("*.rs".into()));
        let call = ToolCall { name: "find_references".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert_eq!(result.value["reference_count"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_find_references_max_results() {
        let dir = TempDir::new().unwrap();
        let content: String = (0..50).map(|_| "target\n").collect();
        fs::write(dir.path().join("big.rs"), content).unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = FindReferencesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("target".into()));
        args.insert("max_results".to_string(), Value::Number(10.into()));
        let call = ToolCall { name: "find_references".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert_eq!(result.value["reference_count"].as_u64().unwrap(), 10);
        assert!(result.value["truncated"].as_bool().unwrap());
    }

    #[test]
    fn test_find_references_no_results() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "fn unrelated() {}").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = FindReferencesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("nonexistent".into()));
        let call = ToolCall { name: "find_references".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["reference_count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_find_references_missing_symbol() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = FindReferencesTool;
        let call = ToolCall {
            name: "find_references".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_find_references_definition() {
        let tool = FindReferencesTool;
        let def = tool.definition();
        assert_eq!(def.name, "find_references");
        assert_eq!(def.required_params(), vec!["symbol"]);
    }

    #[test]
    fn test_find_references_summarize() {
        let tool = FindReferencesTool;
        let result = ToolResult::success(serde_json::json!({
            "symbol": "foo",
            "reference_count": 2,
            "references": [
                {"file": "a.rs", "line": 1, "symbol": "foo", "text": "foo()"},
                {"file": "b.rs", "line": 5, "symbol": "foo", "text": "foo()"},
            ]
        }));
        let summary = tool.summarize_result(&result);
        assert!(summary.contains("Found 2 references"));
        assert!(summary.contains("a.rs:1"));
    }
}
