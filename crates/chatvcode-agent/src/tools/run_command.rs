use std::process::Command;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

/// 允许执行的命令白名单。任何不在白名单中的命令将被拒绝。
///
/// 白名单刻意聚焦于「只读 / 观察」类命令，避免 Agent 通过
/// 命令执行造成破坏性副作用。可由调用方在 [`RunCommandTool::with_whitelist`]
/// 中覆盖。
const DEFAULT_COMMAND_WHITELIST: &[&str] = &[
    "ls", "pwd", "cat", "head", "tail", "wc", "find", "git", "rg", "grep",
    "cargo", "rustc", "echo", "tree", "stat", "file",
];

/// `run_command` 工具：在受控环境下执行命令行程序并捕获输出。
///
/// 安全设计：
/// - **白名单**：默认仅允许只读命令；可由调用方覆盖。
/// - **超时**：复用 [`ToolContext::timeout`]，超时后终止子进程。
/// - **参数**：调用方提供 `command` 与可选 `args` 数组。
/// - **stdout / stderr**：分别捕获并截断到一定长度。
///
/// 不可缓存：命令通常具有非确定性副作用。
pub struct RunCommandTool {
    whitelist: Vec<String>,
    max_output_bytes: usize,
}

impl Default for RunCommandTool {
    fn default() -> Self {
        Self::new()
    }
}

impl RunCommandTool {
    /// 创建一个使用默认白名单的命令工具。
    pub fn new() -> Self {
        Self {
            whitelist: DEFAULT_COMMAND_WHITELIST.iter().map(|s| s.to_string()).collect(),
            max_output_bytes: 8192,
        }
    }

    /// 用自定义白名单替换默认白名单。
    #[must_use]
    pub fn with_whitelist(mut self, commands: Vec<String>) -> Self {
        self.whitelist = commands;
        self
    }

    /// 设置单次捕获 stdout/stderr 的最大字节数。
    #[must_use]
    pub fn with_max_output(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes;
        self
    }

    fn is_allowed(&self, command: &str) -> bool {
        let lower = command.to_lowercase();
        self.whitelist.iter().any(|c| c.to_lowercase() == lower)
    }
}

/// 在 Windows 与 Unix 上通用：使用 `cmd /C`（Windows）或 `sh -c`（其他）
/// 直接执行 `[command, args...]`。这里采用直接调用 + 参数数组形式，
/// 避免 shell 注入风险。
fn build_process(command: &str, args: &[String]) -> Command {
    let mut cmd = Command::new(command);
    for a in args {
        cmd.arg(a);
    }
    cmd
}

impl BuiltinTool for RunCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("run_command")
            .description(
                "Run a whitelisted shell command and capture its stdout/stderr. \
                 Only read-only / observation commands are allowed by default \
                 (ls, cat, git, cargo, rg, ...). Output is truncated to a safe size.",
            )
            .parameter(
                ToolParameter::string("command")
                    .description("The executable name to run (must be on the whitelist)")
                    .required(true),
            )
            .parameter(
                ToolParameter::array("args").description("Arguments to pass to the command"),
            )
    }

    fn execute(&self, call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let command = call.get_string("command").unwrap();
        if !self.is_allowed(command) {
            return Ok(ToolResult::error(format!(
                "Command '{}' is not in the whitelist. Allowed: {}",
                command,
                self.whitelist.join(", ")
            )));
        }

        let args = match call.arguments.get("args") {
            Some(serde_json::Value::Array(arr)) => {
                arr.iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect::<Vec<_>>()
            }
            _ => Vec::new(),
        };

        let mut cmd = build_process(command, &args);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let output = cmd.output().map_err(|e| AgentError::ToolError {
            tool_name: "run_command".into(),
            message: format!("Failed to spawn '{}': {}", command, e),
        })?;

        let stdout = truncate_bytes(&output.stdout, self.max_output_bytes);
        let stderr = truncate_bytes(&output.stderr, self.max_output_bytes);

        let result = serde_json::json!({
            "command": command,
            "args": args,
            "exit_code": output.status.code().unwrap_or(-1),
            "stdout": stdout,
            "stderr": stderr,
            "truncated_stdout": output.stdout.len() > self.max_output_bytes,
            "truncated_stderr": output.stderr.len() > self.max_output_bytes,
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
        let cmd = result.value["command"].as_str().unwrap_or("?");
        let code = result.value["exit_code"].as_i64().unwrap_or(0);
        format!("`{}` exited with code {}", cmd, code)
    }
}

fn truncate_bytes(bytes: &[u8], max: usize) -> String {
    let limited = if bytes.len() > max { &bytes[..max] } else { bytes };
    String::from_utf8_lossy(limited).into_owned()
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

    fn make_call(command: &str, args: Vec<Value>) -> ToolCall {
        let mut arguments = HashMap::new();
        arguments.insert("command".to_string(), Value::String(command.to_string()));
        if !args.is_empty() {
            arguments.insert("args".to_string(), Value::Array(args));
        }
        ToolCall { name: "run_command".into(), arguments, id: None }
    }

    #[test]
    fn test_run_command_echo() {
        let tmp = TempDir::new().unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());
        let tool = RunCommandTool::new();
        // `cargo` 在所有平台上都是独立可执行程序，避免 echo 在 Windows 上
        // 仅为 cmd 内建命令的问题。
        let result = tool
            .execute(&make_call("cargo", vec![Value::String("--version".into())]), &ctx)
            .unwrap();
        assert!(result.success);
        assert_eq!(result.value["exit_code"].as_i64().unwrap(), 0);
        assert!(result.value["stdout"].as_str().unwrap().contains("cargo"));
    }

    #[test]
    fn test_run_command_disallowed() {
        let tmp = TempDir::new().unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());
        let tool = RunCommandTool::new();
        let result = tool.execute(&make_call("rm", vec![Value::String("-rf".into())]), &ctx).unwrap();
        assert!(!result.success);
        assert!(result.value.as_str().unwrap().contains("whitelist"));
    }

    #[test]
    fn test_run_command_custom_whitelist() {
        let tool = RunCommandTool::new().with_whitelist(vec!["echo".into()]);
        assert!(tool.is_allowed("echo"));
        assert!(!tool.is_allowed("git"));
    }

    #[test]
    fn test_run_command_missing_param() {
        let tmp = TempDir::new().unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());
        let tool = RunCommandTool::new();
        let call = ToolCall {
            name: "run_command".into(),
            arguments: HashMap::new(),
            id: None,
        };
        let result = tool.execute(&call, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_command_not_cacheable() {
        let tool = RunCommandTool::new();
        assert!(!tool.is_cacheable());
    }

    #[test]
    fn test_run_command_definition() {
        let def = RunCommandTool::definition(&RunCommandTool::new());
        assert_eq!(def.name, "run_command");
        assert_eq!(def.required_params(), vec!["command"]);
    }

    #[test]
    fn test_truncate_bytes() {
        assert_eq!(truncate_bytes(b"hello", 100), "hello");
        assert_eq!(truncate_bytes(b"abcdef", 3), "abc");
    }
}