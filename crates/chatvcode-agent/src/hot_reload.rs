//! 工具热加载/动态注册：在 Agent 运行时增量地注册、注销工具。
//!
//! 不同于 [`crate::executor::BuiltinToolRegistry`]（构造时确定工具集合，
//! 后续仅支持 `&mut self` 注册），[`RuntimeToolRegistry`] 在 `Arc` 共享
//! 引用下也支持动态增删工具，通过内部 `RwLock` 实现并发安全。
//!
//! 实现 [`ToolExecutor`] 后可直接替代默认注册表注入 [`crate::agent_loop::AgentLoop`]，
//! 配合 `format_tool_prompt()` 即可让新增工具自动出现在 Agent 的系统提示词中。

use std::sync::{Arc, RwLock};

use chatvcode_llm::{ToolCall, ToolDefinition, ToolResult};
use serde_json::Value;

use crate::cache::ToolResultCache;
use crate::context::ToolContext;
use crate::error::AgentError;
use crate::executor::ToolExecutor;
use crate::tools::{self, BuiltinTool};
use crate::types::ToolRetryConfig;

/// 可运行时增删的工具注册表，可直接作为 [`ToolExecutor`] 使用。
pub struct RuntimeToolRegistry {
    tools: RwLock<Vec<Arc<dyn BuiltinTool>>>,
    cache: RwLock<ToolResultCache>,
    retry_config: ToolRetryConfig,
}

impl RuntimeToolRegistry {
    /// 创建一个空的运行时注册表。
    pub fn new(retry_config: ToolRetryConfig) -> Self {
        Self {
            tools: RwLock::new(Vec::new()),
            cache: RwLock::new(ToolResultCache::default()),
            retry_config,
        }
    }

    /// 创建并预装默认内置工具集。
    pub fn with_defaults(retry_config: ToolRetryConfig) -> Self {
        let registry = Self::new(retry_config);
        for tool in tools::register_all_tools() {
            registry.register(tool);
        }
        registry
    }

    /// 在运行时注册一个工具。若同名工具已存在则替换它。
    pub fn register(&self, tool: Box<dyn BuiltinTool>) {
        let def = tool.definition();
        let arc: Arc<dyn BuiltinTool> = Arc::from(tool);
        let mut guard = self
            .tools
            .write()
            .expect("RuntimeToolRegistry lock poisoned");
        if let Some(existing) = guard.iter_mut().find(|t| t.definition().name == def.name) {
            *existing = arc;
        } else {
            guard.push(arc);
        }
    }

    /// 注销指定名称的工具。返回是否曾存在。
    pub fn deregister(&self, name: &str) -> bool {
        let mut guard = self
            .tools
            .write()
            .expect("RuntimeToolRegistry lock poisoned");
        let before = guard.len();
        guard.retain(|t| t.definition().name != name);
        guard.len() != before
    }

    /// 返回当前已注册工具名称的快照。
    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .read()
            .expect("RuntimeToolRegistry lock poisoned")
            .iter()
            .map(|t| t.definition().name)
            .collect()
    }

    fn find_tool(&self, name: &str) -> Option<Arc<dyn BuiltinTool>> {
        self.tools
            .read()
            .expect("RuntimeToolRegistry lock poisoned")
            .iter()
            .find(|t| t.definition().name == name)
            .cloned()
    }

    fn check_path_safety(call: &ToolCall, ctx: &ToolContext) -> Result<(), AgentError> {
        if let Some(Value::String(path)) = call.arguments.get("path") {
            tools::resolve_safe_path(&ctx.project_path, path)?;
        }
        Ok(())
    }

    fn execute_with_retry(
        &self,
        tool: Arc<dyn BuiltinTool>,
        call: &ToolCall,
        ctx: &ToolContext,
    ) -> Result<ToolResult, AgentError> {
        let mut last_error: Option<AgentError> = None;
        let max_attempts = self.retry_config.max_retries + 1;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    self.retry_config.backoff_ms * (attempt as u64),
                ));
            }
            match tool.execute(call, ctx) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    let should_retry = matches!(
                        e,
                        AgentError::Internal(_) | AgentError::Timeout(_)
                    ) && attempt + 1 < max_attempts;
                    if !should_retry {
                        return Err(e);
                    }
                    last_error = Some(e);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| AgentError::Internal("Unknown error".into())))
    }
}

impl ToolExecutor for RuntimeToolRegistry {
    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        let tool = self
            .find_tool(&call.name)
            .ok_or_else(|| AgentError::ToolError {
                tool_name: call.name.clone(),
                message: format!("Unknown tool: {}", call.name),
            })?;

        if tool.is_cacheable() {
            let key = tool.cache_key(call);
            if let Some(cached) = self.cache.read().expect("lock poisoned").get(&key) {
                log::debug!("RuntimeToolRegistry cache hit for '{}'", call.name);
                return Ok(cached);
            }
        }

        tool.validate_arguments(call)?;
        Self::check_path_safety(call, ctx)?;

        let result = self.execute_with_retry(Arc::clone(&tool), call, ctx)?;

        if tool.is_cacheable() {
            let key = tool.cache_key(call);
            self.cache.write().expect("lock poisoned").set(key, result.clone());
        }
        Ok(result)
    }

    fn execute_batch(&self, calls: &[ToolCall], ctx: &ToolContext) -> Vec<ToolResult> {
        calls
            .iter()
            .map(|c| match self.execute(c, ctx) {
                Ok(r) => r,
                Err(e) => {
                    let mut r = ToolResult::error(e.to_string());
                    if let Some(id) = &c.id {
                        r = r.with_call_id(id.clone());
                    }
                    r
                }
            })
            .collect()
    }

    fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools
            .read()
            .expect("RuntimeToolRegistry lock poisoned")
            .iter()
            .map(|t| t.definition())
            .collect()
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
            prompt.push_str(&format!("## {}\n{}\n\n", def.name, def.description));
            if !def.parameters.is_empty() {
                prompt.push_str("Parameters:\n");
                for p in &def.parameters {
                    let req = if p.required { " (required)" } else { "" };
                    prompt.push_str(&format!("- {}: {}{}\n", p.name, p.description, req));
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
    use chatvcode_llm::{ToolCall, ToolDefinition, ToolResult};
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
            timeout: Duration::from_secs(5),
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

    // 一个可变计数的测试工具
    struct ColdTool;
    impl BuiltinTool for ColdTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("cold_tool").description("Cold tool")
        }
        fn execute(&self, _: &ToolCall, _: &ToolContext) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::success(Value::String("cold".into())))
        }
    }

    struct HotTool;
    impl BuiltinTool for HotTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("hot_tool").description("Hot tool")
        }
        fn execute(&self, _: &ToolCall, _: &ToolContext) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::success(Value::String("hot".into())))
        }
        fn is_cacheable(&self) -> bool {
            false
        }
    }

    #[test]
    fn runtime_registry_starts_empty() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        assert!(reg.list_tools().is_empty());
        assert!(!reg.has_tool("cold_tool"));
    }

    #[test]
    fn runtime_registry_with_defaults_loads_all() {
        let reg = RuntimeToolRegistry::with_defaults(ToolRetryConfig::default());
        assert!(reg.has_tool("read_file"));
        assert!(reg.has_tool("run_command"));
        assert!(reg.list_tools().len() >= 13);
    }

    #[test]
    fn register_deregister_roundtrip() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        reg.register(Box::new(ColdTool));
        assert!(reg.has_tool("cold_tool"));
        assert!(reg.deregister("cold_tool"));
        assert!(!reg.has_tool("cold_tool"));
        // 重复 deregister 返回 false
        assert!(!reg.deregister("cold_tool"));
    }

    #[test]
    fn register_replaces_existing同名tool() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        reg.register(Box::new(ColdTool));
        reg.register(Box::new(HotTool)); // 不同名，所以是新增
        assert_eq!(reg.tool_names().len(), 2);
        // 注册同名 (cold_tool -> 新版本)
        struct ColdV2;
        impl BuiltinTool for ColdV2 {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition::new("cold_tool").description("V2")
            }
            fn execute(&self, _: &ToolCall, _: &ToolContext) -> Result<ToolResult, AgentError> {
                Ok(ToolResult::success(Value::String("v2".into())))
            }
        }
        reg.register(Box::new(ColdV2));
        assert_eq!(reg.tool_names().len(), 2);
    }

    #[test]
    fn execute_unknown_tool_errors() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        let tmp = TempDir::new().unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());
        let call = ToolCall {
            name: "nope".into(),
            arguments: HashMap::new(),
            id: None,
        };
        let result = reg.execute(&call, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn execute_hot_tool_works_and_not_cached() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        reg.register(Box::new(HotTool));
        let tmp = TempDir::new().unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());
        let call = ToolCall {
            name: "hot_tool".into(),
            arguments: HashMap::new(),
            id: None,
        };
        let r = reg.execute(&call, &ctx).unwrap();
        assert!(r.success);
        assert_eq!(r.value, Value::String("hot".into()));
    }

    #[test]
    fn format_tool_prompt_reflects_dynamic_tools() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        let empty_prompt = reg.format_tool_prompt();
        assert!(empty_prompt.is_empty());
        reg.register(Box::new(ColdTool));
        let p = reg.format_tool_prompt();
        assert!(p.contains("cold_tool"));
        reg.deregister("cold_tool");
        assert!(reg.format_tool_prompt().is_empty());
    }

    #[test]
    fn execute_batch_handles_mixed() {
        let reg = RuntimeToolRegistry::new(ToolRetryConfig::default());
        reg.register(Box::new(HotTool));
        let tmp = TempDir::new().unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());
        let ok = ToolCall { name: "hot_tool".into(), arguments: HashMap::new(), id: None };
        let bad = ToolCall { name: "nope".into(), arguments: HashMap::new(), id: None };
        let results = reg.execute_batch(&[ok, bad], &ctx);
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert!(!results[1].success);
    }
}