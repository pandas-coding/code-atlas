use std::fs;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};

use crate::context::ToolContext;
use crate::error::AgentError;

use super::{BuiltinTool, resolve_safe_path};

/// `edit_file` 工具：在已有文件中做局部文本替换或行范围替换。
///
/// 支持两种编辑模式（互斥）：
/// 1. `find` + `replace` 字符串替换（仅替换首次匹配）。
/// 2. `offset` + `limit` 行范围替换为 `content`（1-indexed 行号）。
///
/// 采用原子写入（write-to-temp + rename）。失败时原文件保持不变。
/// 不可缓存：写操作具有副作用。
pub struct EditFileTool;

impl BuiltinTool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("edit_file")
            .description(
                "Edit an existing file by either replacing a substring (`find` -> `replace`, \
                 first occurrence only) or substituting a line range (`offset`/`limit`) with new \
                 `content`. Performs an atomic write and fails if the file does not exist.",
            )
            .parameter(
                ToolParameter::string("path")
                    .description("Path to the file to edit (relative to project root)")
                    .required(true),
            )
            .parameter(
                ToolParameter::string("find")
                    .description("Substring to search for and replace with `replace`. \
                                 Mutually exclusive with offset/limit."),
            )
            .parameter(
                ToolParameter::string("replace")
                    .description("Replacement string for the `find` match."),
            )
            .parameter(
                ToolParameter::integer("offset")
                    .description(
                        "Starting line number for a line-range replacement (1-indexed). \
                         Mutually exclusive with find/replace.",
                    ),
            )
            .parameter(
                ToolParameter::integer("limit")
                    .description("Number of lines to replace starting at `offset`. \
                                  Defaults to 1. Use 0 to insert without removing any lines."),
            )
            .parameter(
                ToolParameter::string("content")
                    .description(
                        "New content to substitute at the edited location. \
                         For find/replace mode this is unused (use `replace`).",
                    ),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let path = call.get_string("path").unwrap();
        let resolved = resolve_safe_path(&ctx.project_path, path)?;

        if !resolved.is_file() {
            return Ok(ToolResult::error(format!("File does not exist: {}", path)));
        }

        let original = fs::read_to_string(&resolved).map_err(|e| AgentError::ToolError {
            tool_name: "edit_file".into(),
            message: format!("Failed to read '{}': {}", path, e),
        })?;

        let has_find = call.arguments.contains_key("find");
        let has_offset = call.arguments.contains_key("offset");

        let new_content = if has_find {
            let find = call.get_string("find").unwrap_or_default();
            let replace = call.get_string("replace").unwrap_or_default();
            if find.is_empty() {
                return Ok(ToolResult::error("`find` must be a non-empty string"));
            }
            match original.find(find) {
                Some(idx) => {
                    let mut updated = String::with_capacity(original.len() - find.len() + replace.len());
                    updated.push_str(&original[..idx]);
                    updated.push_str(replace);
                    updated.push_str(&original[idx + find.len()..]);
                    updated
                }
                None => {
                    return Ok(ToolResult::error(format!(
                        "Substring not found in '{}'",
                        path
                    )));
                }
            }
        } else if has_offset {
            let offset = call.get_i64("offset").unwrap_or(1).max(1) as usize;
            let limit = call.get_i64("limit").unwrap_or(1).max(0) as usize;
            let content = call.get_string("content").unwrap_or_default();

            let lines: Vec<&str> = original.split_inclusive('\n').collect();
            if offset > lines.len() + 1 {
                return Ok(ToolResult::error(format!(
                    "offset {} is beyond end of file ({} lines)",
                    offset,
                    lines.len()
                )));
            }

            let mut updated = String::new();
            // 前 offset-1 行
            for line in lines.iter().take(offset - 1) {
                updated.push_str(line);
            }
            // 注入新内容
            if !content.is_empty() {
                if !content.ends_with('\n') {
                    updated.push_str(content);
                    updated.push('\n');
                } else {
                    updated.push_str(content);
                }
            }
            // 跳过最多 limit 行
            let skip_until = (offset - 1 + limit).min(lines.len());
            for line in lines.iter().skip(skip_until) {
                updated.push_str(line);
            }
            updated
        } else {
            return Ok(ToolResult::error(
                "Either `find`+`replace` or `offset` (+`limit`) must be provided",
            ));
        };

        atomic_write(&resolved, &new_content).map_err(|e| AgentError::ToolError {
            tool_name: "edit_file".into(),
            message: format!("Failed to write edited file '{}': {}", path, e),
        })?;

        let result = serde_json::json!({
            "path": path,
            "original_bytes": original.len(),
            "new_bytes": new_content.len(),
            "original_lines": original.lines().count(),
            "new_lines": new_content.lines().count(),
        });
        Ok(ToolResult::success(result))
    }

    fn is_cacheable(&self) -> bool {
        false
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            return format!("Error: {}", result.value);
        }
        let path = result.value["path"].as_str().unwrap_or("?");
        let ob = result.value["original_bytes"].as_u64().unwrap_or(0);
        let nb = result.value["new_bytes"].as_u64().unwrap_or(0);
        format!("Edited {} ({} -> {} bytes)", path, ob, nb)
    }
}

fn atomic_write(target: &std::path::Path, content: &str) -> std::io::Result<()> {
    let tmp = target.with_extension(
        target
            .extension()
            .map(|e| format!("{}.tmp", e.to_string_lossy()))
            .unwrap_or_else(|| "tmp".to_string()),
    );
    fs::write(&tmp, content)?;
    fs::rename(&tmp, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreTrait, CodeSearchService};
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use serde_json::Value;
    use std::collections::HashMap;
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

    fn make_call(args: HashMap<String, Value>) -> ToolCall {
        ToolCall { name: "edit_file".into(), arguments: args, id: None }
    }

    #[test]
    fn test_edit_file_find_replace() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "hello world").unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("a.txt".into()));
        args.insert("find".to_string(), Value::String("world".into()));
        args.insert("replace".to_string(), Value::String("rust".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx).unwrap();
        assert!(result.success);
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "hello rust");
    }

    #[test]
    fn test_edit_file_find_not_found() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "hello").unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("a.txt".into()));
        args.insert("find".to_string(), Value::String("missing".into()));
        args.insert("replace".to_string(), Value::String("x".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx).unwrap();
        assert!(!result.success);
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "hello");
    }

    #[test]
    fn test_edit_file_line_range_replace() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "l1\nl2\nl3\nl4\nl5\n").unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("a.txt".into()));
        args.insert("offset".to_string(), Value::Number(2.into()));
        args.insert("limit".to_string(), Value::Number(2.into()));
        args.insert("content".to_string(), Value::String("REPLACED".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx).unwrap();
        assert!(result.success);
        let after = fs::read_to_string(dir.path().join("a.txt")).unwrap();
        // offset=2, limit=2 -> 替换 l2 与 l3，保留 l4 l5
        assert_eq!(after, "l1\nREPLACED\nl4\nl5\n");
    }

    #[test]
    fn test_edit_file_line_range_insert() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "l1\nl2\n").unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("a.txt".into()));
        args.insert("offset".to_string(), Value::Number(1.into()));
        args.insert("limit".to_string(), Value::Number(0.into()));
        args.insert("content".to_string(), Value::String("INSERTED".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx).unwrap();
        assert!(result.success);
        let after = fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert_eq!(after, "INSERTED\nl1\nl2\n");
    }

    #[test]
    fn test_edit_file_missing_file() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("nonexistent.txt".into()));
        args.insert("find".to_string(), Value::String("x".into()));
        args.insert("replace".to_string(), Value::String("y".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_edit_file_no_mode_specified() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "x").unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("a.txt".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_edit_file_path_violation() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("../../out.txt".into()));
        args.insert("find".to_string(), Value::String("x".into()));
        args.insert("replace".to_string(), Value::String("y".into()));
        let result = EditFileTool.execute(&make_call(args), &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_edit_file_not_cacheable() {
        assert!(!EditFileTool.is_cacheable());
    }

    #[test]
    fn test_edit_file_definition() {
        let def = EditFileTool.definition();
        assert_eq!(def.name, "edit_file");
        assert_eq!(def.required_params(), vec!["path"]);
    }
}