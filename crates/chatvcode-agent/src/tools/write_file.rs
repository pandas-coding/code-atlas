use std::fs;
use std::io::Write;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};

use crate::context::ToolContext;
use crate::error::AgentError;

use super::{BuiltinTool, resolve_safe_path_for_write};

/// `write_file` 工具：将整段内容写入文件，必要时创建父目录。
///
/// 采用「先写临时文件再原子重命名」策略，确保写入要么完整成功，要么
/// 不改变原文件内容。专属工具，区别于 [`EditFileTool`]（局部编辑）。
///
/// 不可缓存：写操作具有副作用，重复调用语义不同。
pub struct WriteFileTool;

impl BuiltinTool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("write_file")
            .description(
                "Write content to a file, creating the file and parent directories \
                 if needed. Overwrites any existing content. Performs an atomic \
                 write (write to temp file, then rename).",
            )
            .parameter(
                ToolParameter::string("path")
                    .description("Path to the file to write (relative to project root)")
                    .required(true),
            )
            .parameter(
                ToolParameter::string("content")
                    .description("Full content to write to the file")
                    .required(true),
            )
            .parameter(
                ToolParameter::boolean("create_dirs")
                    .description("Create parent directories if they do not exist (default: true)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let path = call.get_string("path").unwrap();
        let content = call.get_string("content").unwrap_or_default();
        let create_dirs = call.get_bool("create_dirs").unwrap_or(true);

        let resolved = resolve_safe_path_for_write(&ctx.project_path, path)?;

        if let Some(parent) = resolved.parent()
            && !parent.exists()
        {
            if !create_dirs {
                return Ok(ToolResult::error(format!(
                    "Parent directory does not exist: {} (set create_dirs=true to create it)",
                    parent.display()
                )));
            }
            fs::create_dir_all(parent).map_err(|e| AgentError::ToolError {
                tool_name: "write_file".into(),
                message: format!("Failed to create parent directories for '{}': {}", path, e),
            })?;
        }

        // 原子写入：先写入同目录下的临时文件，再 rename 到目标路径。
        // 同目录 rename 在同一文件系统上是原子的。
        let tmp_path = resolved.with_extension(
            resolved
                .extension()
                .map(|e| format!("{}.tmp", e.to_string_lossy()))
                .unwrap_or_else(|| "tmp".to_string()),
        );
        {
            let mut file = fs::File::create(&tmp_path).map_err(|e| AgentError::ToolError {
                tool_name: "write_file".into(),
                message: format!("Failed to create temp file for '{}': {}", path, e),
            })?;
            file.write_all(content.as_bytes()).map_err(|e| AgentError::ToolError {
                tool_name: "write_file".into(),
                message: format!("Failed to write content to '{}': {}", path, e),
            })?;
            file.sync_all().ok();
        }

        fs::rename(&tmp_path, &resolved).map_err(|e| {
            // rename 失败时清理临时文件
            let _ = fs::remove_file(&tmp_path);
            AgentError::ToolError {
                tool_name: "write_file".into(),
                message: format!("Failed to atomic-rename temp file to '{}': {}", path, e),
            }
        })?;

        let bytes_written = content.len();
        let lines_written = content.lines().count();

        let result = serde_json::json!({
            "path": path,
            "bytes_written": bytes_written,
            "lines_written": lines_written,
            "created": !resolved.exists(), // best-effort hint (already-overwritten)
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
        let bytes = result.value["bytes_written"].as_u64().unwrap_or(0);
        let lines = result.value["lines_written"].as_u64().unwrap_or(0);
        let path = result.value["path"].as_str().unwrap_or("?");
        format!("Wrote {} bytes ({} lines) to {}", bytes, lines, path)
    }
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
        ToolCall { name: "write_file".into(), arguments: args, id: None }
    }

    #[test]
    fn test_write_file_creates_new() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = WriteFileTool;

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("hello.txt".into()));
        args.insert("content".to_string(), Value::String("line1\nline2\n".into()));
        let result = tool.execute(&make_call(args), &ctx).unwrap();
        assert!(result.success);
        let written = fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert_eq!(written, "line1\nline2\n");
        assert_eq!(result.value["bytes_written"].as_u64().unwrap(), 12);
        assert_eq!(result.value["lines_written"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_write_file_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = WriteFileTool;

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("sub/dir/hello.txt".into()));
        args.insert("content".to_string(), Value::String("nested".into()));
        let result = tool.execute(&make_call(args), &ctx).unwrap();
        assert!(result.success);
        assert!(dir.path().join("sub/dir/hello.txt").is_file());
    }

    #[test]
    fn test_write_file_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "old").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = WriteFileTool;
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("a.txt".into()));
        args.insert("content".to_string(), Value::String("new".into()));
        let result = tool.execute(&make_call(args), &ctx).unwrap();
        assert!(result.success);
        assert_eq!(fs::read_to_string(dir.path().join("a.txt")).unwrap(), "new");
    }

    #[test]
    fn test_write_file_no_create_dirs_when_disabled() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = WriteFileTool;
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("missing_dir/x.txt".into()));
        args.insert("content".to_string(), Value::String("x".into()));
        args.insert("create_dirs".to_string(), Value::Bool(false));
        let result = tool.execute(&make_call(args), &ctx).unwrap();
        assert!(!result.success);
        assert!(result.value.as_str().unwrap().contains("Parent directory"));
    }

    #[test]
    fn test_write_file_path_violation() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = WriteFileTool;
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("../../../etc/exploit".into()));
        args.insert("content".to_string(), Value::String("x".into()));
        let result = tool.execute(&make_call(args), &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_file_missing_params() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = WriteFileTool;
        let result = tool.execute(&make_call(HashMap::new()), &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_file_not_cacheable() {
        assert!(!WriteFileTool.is_cacheable());
    }

    #[test]
    fn test_write_file_definition() {
        let def = WriteFileTool.definition();
        assert_eq!(def.name, "write_file");
        assert_eq!(def.required_params(), vec!["path", "content"]);
    }
}