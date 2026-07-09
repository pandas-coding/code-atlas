use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolResult};
use serde_json::Value;

use crate::cache::ToolResultCache;
use crate::context::ToolContext;
use crate::error::AgentError;
use crate::tools::{
    self, BuiltinTool, CompareFilesTool, EditFileTool, FindReferencesTool,
    GetDependenciesTool, GetFileStructureTool, GetProjectOverviewTool, GrepCodeTool,
    ListFilesTool, ReadFileTool, RunCommandTool, SearchCodeTool, SearchSymbolTool,
    WriteFileTool,
};
use crate::types::ToolRetryConfig;

pub trait ToolExecutor: Send + Sync {
    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError>;
    fn execute_batch(&self, calls: &[ToolCall], ctx: &ToolContext) -> Vec<ToolResult>;
    fn list_tools(&self) -> Vec<ToolDefinition>;
    fn has_tool(&self, name: &str) -> bool;
    fn format_tool_prompt(&self) -> String;
}

pub struct BuiltinToolRegistry {
    tools: Vec<Arc<dyn BuiltinTool>>,
    cache: ToolResultCache,
    retry_config: ToolRetryConfig,
}

impl BuiltinToolRegistry {
    pub fn new(retry_config: ToolRetryConfig) -> Self {
        Self { tools: Vec::new(), cache: ToolResultCache::default(), retry_config }
    }

    pub fn register(&mut self, tool: Box<dyn BuiltinTool>) {
        self.tools.push(Arc::from(tool));
    }

    pub fn register_defaults(&mut self) {
        self.tools.push(Arc::new(ReadFileTool));
        self.tools.push(Arc::new(ListFilesTool));
        self.tools.push(Arc::new(GrepCodeTool));
        self.tools.push(Arc::new(GetFileStructureTool));
        self.tools.push(Arc::new(SearchSymbolTool));
        self.tools.push(Arc::new(SearchCodeTool));
        self.tools.push(Arc::new(FindReferencesTool));
        self.tools.push(Arc::new(GetDependenciesTool));
        self.tools.push(Arc::new(CompareFilesTool));
        self.tools.push(Arc::new(GetProjectOverviewTool));
        self.tools.push(Arc::new(WriteFileTool));
        self.tools.push(Arc::new(EditFileTool));
        self.tools.push(Arc::new(RunCommandTool::new()));
    }

    fn find_tool(&self, name: &str) -> Option<Arc<dyn BuiltinTool>> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .cloned()
    }

    fn check_path_safety(&self, call: &ToolCall, ctx: &ToolContext) -> Result<(), AgentError> {
        if let Some(Value::String(path)) = call.arguments.get("path") {
            tools::resolve_safe_path(&ctx.project_path, path)?;
        }
        Ok(())
    }

    fn execute_with_timeout(
        tool: Arc<dyn BuiltinTool>,
        call: ToolCall,
        ctx: ToolContext,
        timeout: Duration,
    ) -> Result<ToolResult, AgentError> {
        let (tx, rx) = mpsc::channel();
        let tool_name = call.name.clone();

        let handle = thread::spawn(move || {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| tool.execute(&call, &ctx)));
            let _ = tx.send(result);
        });

        match rx.recv_timeout(timeout) {
            Ok(Ok(result)) => {
                let _ = handle.join();
                result
            }
            Ok(Err(_panic)) => {
                let _ = handle.join();
                Err(AgentError::Internal(format!("Tool '{}' panicked during execution", tool_name)))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Err(AgentError::Timeout(timeout.as_secs())),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = handle.join();
                Err(AgentError::Internal("Tool execution thread disconnected".into()))
            }
        }
    }

    fn execute_with_retry(
        &self,
        tool: Arc<dyn BuiltinTool>,
        call: &ToolCall,
        ctx: &ToolContext,
    ) -> Result<ToolResult, AgentError> {
        let mut last_error = None;
        let max_attempts = self.retry_config.max_retries + 1;

        for attempt in 0..max_attempts {
            if attempt > 0 {
                let backoff =
                    Duration::from_millis(self.retry_config.backoff_ms * (attempt as u64));
                thread::sleep(backoff);
                log::debug!(
                    "Retrying tool '{}' (attempt {}/{})",
                    call.name,
                    attempt + 1,
                    max_attempts
                );
            }

            match Self::execute_with_timeout(
                Arc::clone(&tool),
                call.clone(),
                ctx_clone(ctx),
                ctx.timeout,
            ) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let should_retry = match &e {
                        AgentError::Timeout(_) => {
                            self.retry_config.retry_on_timeout && attempt + 1 < max_attempts
                        }
                        AgentError::Internal(_) => {
                            self.retry_config.retry_on_transient_error && attempt + 1 < max_attempts
                        }
                        _ => false,
                    };

                    if !should_retry {
                        return Err(e);
                    }

                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| AgentError::Internal("Unknown error".into())))
    }

    fn format_result(call: &ToolCall, result: &Result<ToolResult, AgentError>) -> ToolResult {
        match result {
            Ok(r) => r.clone(),
            Err(e) => {
                let mut err_result = ToolResult::error(e.to_string());
                if let Some(ref id) = call.id {
                    err_result = err_result.with_call_id(id.clone());
                }
                err_result
            }
        }
    }
}

fn ctx_clone(ctx: &ToolContext) -> ToolContext {
    ToolContext {
        project_path: ctx.project_path.clone(),
        timeout: ctx.timeout,
        token_budget: ctx.token_budget,
        services: Arc::clone(&ctx.services),
    }
}

impl ToolExecutor for BuiltinToolRegistry {
    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        let tool = self
            .find_tool(&call.name)
            .ok_or_else(|| AgentError::ToolError {
                tool_name: call.name.clone(),
                message: format!("Unknown tool: {}", call.name),
            })?;

        if tool.is_cacheable() {
            let key = tool.cache_key(call);
            if let Some(cached) = self.cache.get(&key) {
                log::debug!("Cache hit for tool '{}'", call.name);
                return Ok(cached);
            }
        }

        tool.validate_arguments(call)?;

        self.check_path_safety(call, ctx)?;

        let result = self.execute_with_retry(Arc::clone(&tool), call, ctx)?;

        if tool.is_cacheable() {
            let key = tool.cache_key(call);
            self.cache.set(key, result.clone());
        }

        Ok(result)
    }

    fn execute_batch(&self, calls: &[ToolCall], ctx: &ToolContext) -> Vec<ToolResult> {
        calls
            .iter()
            .map(|call| {
                let result = self.execute(call, ctx);
                Self::format_result(call, &result)
            })
            .collect()
    }

    fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    fn has_tool(&self, name: &str) -> bool {
        self.find_tool(name).is_some()
    }

    fn format_tool_prompt(&self) -> String {
        let defs = self.list_tools();
        if defs.is_empty() {
            return String::new();
        }

        let mut prompt = String::from(
            "You have access to the following tools. To use a tool, respond with a JSON object in this format:\n\
             {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n\n\
             Available tools:\n\n",
        );

        for def in &defs {
            prompt.push_str(&format!("## {}\n", def.name));
            prompt.push_str(&format!("{}\n\n", def.description));

            if !def.parameters.is_empty() {
                prompt.push_str("Parameters:\n");
                for param in &def.parameters {
                    let required = if param.required { " (required)" } else { "" };
                    prompt.push_str(&format!(
                        "- {}: {}{}\n",
                        param.name, param.description, required
                    ));
                    if !param.enum_values.is_empty() {
                        prompt.push_str(&format!(
                            "  Allowed values: {}\n",
                            param.enum_values.join(", ")
                        ));
                    }
                }
                prompt.push('\n');
            }
        }

        prompt
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
    use std::time::Instant;

    struct MockSearchService;
    impl CodeSearchService for MockSearchService {
        fn search(&self, _query: &str, _top_k: usize) -> Result<Vec<SearchResult>, AgentError> {
            Ok(vec![])
        }
    }

    struct MockChunkStore;
    impl ChunkMetadataStoreTrait for MockChunkStore {
        fn get_chunks_by_symbol(&self, _symbol: &str, _kind: Option<&str>) -> Vec<ChunkMetadata> {
            vec![]
        }
        fn get_chunk_by_id(&self, _id: &str) -> Option<ChunkMetadata> {
            None
        }
    }

    fn make_test_ctx(project_path: PathBuf) -> ToolContext {
        let services =
            Arc::new(
                AgentServices {
                    search: Box::new(MockSearchService),
                    parser:
                        Box::new(
                            |_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<
                                chatvcode_core::model::ParseResult,
                            > { unimplemented!() },
                        ),
                    chunk_store: Box::new(MockChunkStore),
                },
            );
        ToolContext { project_path, timeout: Duration::from_secs(5), token_budget: 4096, services }
    }

    fn make_call(name: &str, args: HashMap<String, Value>) -> ToolCall {
        ToolCall { name: name.to_string(), arguments: args, id: None }
    }

    struct SuccessTool;
    impl BuiltinTool for SuccessTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("success_tool")
                .description("Always succeeds")
                .parameter(
                    chatvcode_llm::ToolParameter::string("input")
                        .description("Input value")
                        .required(true),
                )
        }

        fn execute(&self, call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
            let input = call.get_string("input").unwrap_or("default");
            Ok(ToolResult::success(Value::String(format!("ok: {}", input))))
        }
    }

    struct FailTool;
    impl BuiltinTool for FailTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("fail_tool").description("Always fails")
        }

        fn execute(&self, _call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
            Err(AgentError::ToolError {
                tool_name: "fail_tool".into(),
                message: "intentional failure".into(),
            })
        }
    }

    struct PanicTool;
    impl BuiltinTool for PanicTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("panic_tool").description("Always panics")
        }

        fn execute(&self, _call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
            panic!("intentional panic");
        }
    }

    struct SlowTool;
    impl BuiltinTool for SlowTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("slow_tool").description("Takes a long time")
        }

        fn execute(&self, _call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
            thread::sleep(Duration::from_secs(1));
            Ok(ToolResult::success(Value::String("done".into())))
        }
    }

    struct TransientFailTool {
        call_count: std::sync::atomic::AtomicUsize,
        fail_times: usize,
    }

    impl TransientFailTool {
        fn new(fail_times: usize) -> Self {
            Self { call_count: std::sync::atomic::AtomicUsize::new(0), fail_times }
        }
    }

    impl BuiltinTool for TransientFailTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("transient_tool").description("Fails N times then succeeds")
        }

        fn execute(&self, _call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
            let count = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count < self.fail_times {
                Err(AgentError::Internal(format!("transient error (attempt {})", count + 1)))
            } else {
                Ok(ToolResult::success(Value::String("recovered".into())))
            }
        }
    }

    struct PathTool;
    impl BuiltinTool for PathTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("path_tool")
                .description("A tool that takes a path")
                .parameter(
                    chatvcode_llm::ToolParameter::string("path")
                        .description("File path")
                        .required(true),
                )
        }

        fn execute(&self, call: &ToolCall, _ctx: &ToolContext) -> Result<ToolResult, AgentError> {
            let path = call.get_string("path").unwrap_or("");
            Ok(ToolResult::success(Value::String(format!("read: {}", path))))
        }
    }

    fn make_test_registry() -> BuiltinToolRegistry {
        let mut reg = BuiltinToolRegistry::new(ToolRetryConfig {
            max_retries: 0,
            retry_on_timeout: false,
            retry_on_transient_error: false,
            backoff_ms: 10,
        });
        reg.register(Box::new(SuccessTool));
        reg.register(Box::new(FailTool));
        reg.register(Box::new(PanicTool));
        reg.register(Box::new(SlowTool));
        reg.register(Box::new(PathTool));
        reg
    }

    #[test]
    fn test_execute_normal() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("input".to_string(), Value::String("hello".into()));
        let call = make_call("success_tool", args);

        let result = reg.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value, Value::String("ok: hello".into()));
    }

    #[test]
    fn test_execute_unknown_tool() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let call = make_call("nonexistent_tool", HashMap::new());
        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolError { tool_name, message } => {
                assert_eq!(tool_name, "nonexistent_tool");
                assert!(message.contains("Unknown tool"));
            }
            other => panic!("Expected ToolError, got: {:?}", other),
        }
    }

    #[test]
    fn test_execute_validation_failure() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let call = make_call("success_tool", HashMap::new());
        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolError { message, .. } => {
                assert!(message.contains("Missing required parameter"));
            }
            other => panic!("Expected ToolError, got: {:?}", other),
        }
    }

    #[test]
    fn test_execute_path_violation() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("../../../../etc/passwd".into()));
        let call = make_call("path_tool", args);

        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolError { message, .. } => {
                assert!(message.contains("outside the project directory"));
            }
            other => panic!("Expected ToolError for path violation, got: {:?}", other),
        }
    }

    #[test]
    fn test_execute_timeout() {
        let mut reg = BuiltinToolRegistry::new(ToolRetryConfig {
            max_retries: 0,
            retry_on_timeout: false,
            retry_on_transient_error: false,
            backoff_ms: 10,
        });
        reg.register(Box::new(SlowTool));

        let tmp = tempfile::tempdir().unwrap();
        let mut ctx = make_test_ctx(tmp.path().to_path_buf());
        ctx.timeout = Duration::from_millis(100);

        let call = make_call("slow_tool", HashMap::new());
        let start = Instant::now();
        let result = reg.execute(&call, &ctx);
        let elapsed = start.elapsed();

        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::Timeout(_) => {}
            other => panic!("Expected Timeout, got: {:?}", other),
        }
        assert!(elapsed < Duration::from_secs(5));
    }

    #[test]
    fn test_execute_error() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let call = make_call("fail_tool", HashMap::new());
        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ToolError { tool_name, message } => {
                assert_eq!(tool_name, "fail_tool");
                assert!(message.contains("intentional failure"));
            }
            other => panic!("Expected ToolError, got: {:?}", other),
        }
    }

    #[test]
    fn test_execute_panic_captured() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let call = make_call("panic_tool", HashMap::new());
        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::Internal(msg) => {
                assert!(msg.contains("panicked"));
            }
            other => panic!("Expected Internal error for panic, got: {:?}", other),
        }
    }

    #[test]
    fn test_retry_mechanism() {
        let retry_config = ToolRetryConfig {
            max_retries: 3,
            retry_on_timeout: true,
            retry_on_transient_error: true,
            backoff_ms: 10,
        };
        let mut reg = BuiltinToolRegistry::new(retry_config);
        reg.register(Box::new(TransientFailTool::new(2)));

        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let call = make_call("transient_tool", HashMap::new());
        let result = reg.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value, Value::String("recovered".into()));
    }

    #[test]
    fn test_retry_exhausted() {
        let retry_config = ToolRetryConfig {
            max_retries: 1,
            retry_on_timeout: false,
            retry_on_transient_error: true,
            backoff_ms: 10,
        };
        let mut reg = BuiltinToolRegistry::new(retry_config);
        reg.register(Box::new(TransientFailTool::new(5)));

        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let call = make_call("transient_tool", HashMap::new());
        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_hit() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("input".to_string(), Value::String("cached".into()));
        let call = make_call("success_tool", args);

        let r1 = reg.execute(&call, &ctx).unwrap();
        let r2 = reg.execute(&call, &ctx).unwrap();
        assert_eq!(r1.value, r2.value);
    }

    #[test]
    fn test_execute_batch() {
        let reg = make_test_registry();
        let tmp = tempfile::tempdir().unwrap();
        let ctx = make_test_ctx(tmp.path().to_path_buf());

        let mut args1 = HashMap::new();
        args1.insert("input".to_string(), Value::String("a".into()));
        let call1 = make_call("success_tool", args1);

        let call2 = make_call("fail_tool", HashMap::new());
        let call3 = make_call("unknown", HashMap::new());

        let results = reg.execute_batch(&[call1, call2, call3], &ctx);
        assert_eq!(results.len(), 3);
        assert!(results[0].success);
        assert!(!results[1].success);
        assert!(!results[2].success);
    }

    #[test]
    fn test_list_tools() {
        let reg = make_test_registry();
        let defs = reg.list_tools();
        assert_eq!(defs.len(), 5);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"success_tool"));
        assert!(names.contains(&"fail_tool"));
    }

    #[test]
    fn test_has_tool() {
        let reg = make_test_registry();
        assert!(reg.has_tool("success_tool"));
        assert!(reg.has_tool("fail_tool"));
        assert!(!reg.has_tool("nonexistent"));
    }

    #[test]
    fn test_format_tool_prompt() {
        let reg = make_test_registry();
        let prompt = reg.format_tool_prompt();
        assert!(prompt.contains("Available tools"));
        assert!(prompt.contains("success_tool"));
        assert!(prompt.contains("fail_tool"));
    }

    #[test]
    fn test_register_defaults() {
        let mut reg = BuiltinToolRegistry::new(ToolRetryConfig::default());
        reg.register_defaults();
        assert_eq!(reg.list_tools().len(), 13);
        assert!(reg.has_tool("read_file"));
        assert!(reg.has_tool("list_files"));
        assert!(reg.has_tool("grep_code"));
        assert!(reg.has_tool("get_file_structure"));
        assert!(reg.has_tool("search_symbol"));
        assert!(reg.has_tool("search_code"));
        assert!(reg.has_tool("find_references"));
        assert!(reg.has_tool("get_dependencies"));
        assert!(reg.has_tool("compare_files"));
        assert!(reg.has_tool("get_project_overview"));
        assert!(reg.has_tool("write_file"));
        assert!(reg.has_tool("edit_file"));
        assert!(reg.has_tool("run_command"));
    }

    #[test]
    fn test_empty_registry_format_prompt() {
        let reg = BuiltinToolRegistry::new(ToolRetryConfig::default());
        assert!(reg.format_tool_prompt().is_empty());
    }

    #[test]
    fn test_format_result_error() {
        let call = ToolCall {
            name: "test".into(),
            arguments: HashMap::new(),
            id: Some("call_123".into()),
        };
        let err = Err(AgentError::ToolError { tool_name: "test".into(), message: "bad".into() });
        let result = BuiltinToolRegistry::format_result(&call, &err);
        assert!(!result.success);
        assert_eq!(result.call_id.as_deref(), Some("call_123"));
    }
}
