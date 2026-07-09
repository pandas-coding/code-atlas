//! 非代码库场景的 Agent 扩展：将 Agent 工具集抽象为可插拔「场景」概念，
//! 允许同一套 Agent 循环复用于非代码场景（如文档库、通用文本库）。
//!
//! 每个 [`Scenario`] 描述其名称、适用的工具集以及可选的系统提示词修饰。
//! 调用方通过 [`Scenario::build_tools`] 取得工具集合，注入
//! [`RuntimeToolRegistry`](crate::hot_reload::RuntimeToolRegistry) 或自定义
//! [`ToolExecutor`](crate::executor::ToolExecutor) 即可让 Agent 适配该场景。
//!
//! 当前提供两个开箱即用的场景：
//! - [`DocumentScenario`]：文档库场景（read_text_file、list_docs、search_text）
//! - [`GenericTextScenario`]：通用文本场景，仅提供只读工具的子集

use chatvcode_llm::{ToolCall, ToolDefinition, ToolResult};

use crate::context::ToolContext;
use crate::error::AgentError;
use crate::tools::BuiltinTool;

/// 场景抽象：描述 Agent 在某一类内容场景下的工具集合行为。
pub trait Scenario: Send + Sync {
    /// 场景唯一标识，例如 `"documents"`。
    fn name(&self) -> &str;

    /// 场景的人类可读描述。
    fn description(&self) -> &str;

    /// 构造该场景下可用的工具集合。
    fn build_tools(&self) -> Vec<Box<dyn BuiltinTool>>;

    /// 可选的系统提示词修饰（追加到默认系统提示词后）。
    fn system_prompt_modifier(&self) -> Option<String> {
        None
    }
}

/// 文档库场景：提供基于文本的浏览/检索工具，
/// 适用于 Markdown / 文档目录等非源代码内容。
pub struct DocumentScenario {
    extensions: Vec<String>,
}

impl DocumentScenario {
    /// 创建一个监听指定扩展名的文档场景。
    pub fn new<I: IntoIterator<Item = String>>(extensions: I) -> Self {
        Self { extensions: extensions.into_iter().collect() }
    }

    /// 创建一个常见的 Markdown/txt 文档场景。
    pub fn markdown() -> Self {
        Self::new(["md".to_string(), "markdown".to_string(), "txt".to_string()])
    }
}

impl Default for DocumentScenario {
    fn default() -> Self {
        Self::markdown()
    }
}

impl Scenario for DocumentScenario {
    fn name(&self) -> &str {
        "documents"
    }

    fn description(&self) -> &str {
        "A document library scenario providing tooling for browsing, reading and \
         full-text searching Markdown / text documents."
    }

    fn build_tools(&self) -> Vec<Box<dyn BuiltinTool>> {
        vec![
            Box::new(ReadTextFileTool),
            Box::new(ListDocsTool {
                extensions: self.extensions.clone(),
            }),
            Box::new(SearchTextTool),
        ]
    }

    fn system_prompt_modifier(&self) -> Option<String> {
        Some(
            "You are operating in a document library scenario. Use read_text_file / \
             list_docs / search_text to explore the documents directory."
                .to_string(),
        )
    }
}

/// 通用文本场景：仅提供只读工具的最小子集，适合任意文本资源。
pub struct GenericTextScenario;

impl Scenario for GenericTextScenario {
    fn name(&self) -> &str {
        "generic-text"
    }
    fn description(&self) -> &str {
        "A generic text scenario that exposes only read-only file tools."
    }
    fn build_tools(&self) -> Vec<Box<dyn BuiltinTool>> {
        vec![Box::new(ReadTextFileTool), Box::new(SearchTextTool)]
    }
}

// ---- 内置文档场景工具 --------------------------------------------------

/// 读取文本文件整体内容（不做语法解析）。
pub struct ReadTextFileTool;

impl BuiltinTool for ReadTextFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("read_text_file")
            .description("Read the full text contents of a document file.")
            .parameter(
                chatvcode_llm::ToolParameter::string("path")
                    .description("Path to the document (relative to project root)")
                    .required(true),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;
        let path = call.get_string("path").unwrap();
        let resolved = crate::tools::resolve_safe_path(&ctx.project_path, path)?;
        if !resolved.is_file() {
            return Ok(ToolResult::error(format!("Not a file: {}", path)));
        }
        std::fs::read_to_string(&resolved)
            .map(|content| {
                let lines = content.lines().count();
                ToolResult::success(serde_json::json!({
                    "path": path,
                    "lines": lines,
                    "content": content,
                }))
            })
            .map_err(|e| AgentError::ToolError {
                tool_name: "read_text_file".into(),
                message: format!("Failed to read '{}': {}", path, e),
            })
    }
}

/// 列出文档目录中的文件，按扩展名过滤。
pub struct ListDocsTool {
    extensions: Vec<String>,
}

impl BuiltinTool for ListDocsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("list_docs")
            .description(
                "List document files in the project, optionally filtered by extension.",
            )
            .parameter(
                chatvcode_llm::ToolParameter::string("path")
                    .description("Directory to list (relative to project root; default: project root)"),
            )
            .parameter(
                chatvcode_llm::ToolParameter::integer("max_depth")
                    .description("Maximum recursion depth (default: 10)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        let path = call.get_string("path").unwrap_or(".");
        let max_depth = call.get_i64("max_depth").unwrap_or(10).max(1) as usize;
        let base = crate::tools::resolve_safe_path(&ctx.project_path, path)?;
        if !base.is_dir() {
            return Ok(ToolResult::error(format!("Not a directory: {}", path)));
        }
        let mut files: Vec<String> = Vec::new();
        for entry in walkdir::WalkDir::new(&base)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let p = entry.path();
            let ext_match = match p.extension().and_then(|e| e.to_str()) {
                Some(e) => self.extensions.iter().any(|x| x.eq_ignore_ascii_case(e)),
                None => false,
            };
            if ext_match
                && let Some(rel) = strip_projects(p, &ctx.project_path)
            {
                files.push(rel);
            }
        }
        files.sort();
        Ok(ToolResult::success(serde_json::json!({
            "path": path,
            "count": files.len(),
            "files": files,
        })))
    }
}

/// 在所有文本文件中做纯文本子串搜索（区分大小写）。
pub struct SearchTextTool;

impl BuiltinTool for SearchTextTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("search_text")
            .description(
                "Search for a literal substring across all files in the project and \
                 return file:line matches.",
            )
            .parameter(
                chatvcode_llm::ToolParameter::string("needle")
                    .description("Substring to search for")
                    .required(true),
            )
            .parameter(
                chatvcode_llm::ToolParameter::integer("limit")
                    .description("Maximum number of matches to return (default: 50)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;
        let needle = call.get_string("needle").unwrap();
        let limit = call.get_i64("limit").unwrap_or(50).max(1) as usize;
        let mut matches: Vec<serde_json::Value> = Vec::new();
        for entry in walkdir::WalkDir::new(&ctx.project_path)
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let p = entry.path();
            if let Ok(content) = std::fs::read_to_string(p) {
                for (i, line) in content.lines().enumerate() {
                    if line.contains(needle) {
                        let rel = strip_projects(p, &ctx.project_path)
                            .unwrap_or_else(|| p.to_string_lossy().into_owned());
                        matches.push(serde_json::json!({
                            "path": rel,
                            "line": i + 1,
                            "text": truncate(line, 200),
                        }));
                        if matches.len() >= limit {
                            return Ok(ToolResult::success(serde_json::json!({
                                "needle": needle,
                                "matches": matches,
                                "truncated": true,
                            })));
                        }
                    }
                }
            }
        }
        Ok(ToolResult::success(serde_json::json!({
            "needle": needle,
            "matches": matches,
            "truncated": false,
        })))
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{}...", head)
    }
}

/// 将 `path` 相对 `project` 转换为正斜杠分隔的字符串路径。
///
/// 当 `path` 经过 `walkdir`（规范化路径）而 `project` 未规范化时，先尝试
/// 直接 `strip_prefix`，失败则在 `canonicalize(project)` 与原始路径上重试。
/// 任何一种成功即返回结果；全部失败时返回 `None`。
fn strip_projects(path: &std::path::Path, project: &std::path::Path) -> Option<String> {
    if let Ok(rel) = path.strip_prefix(project) {
        return Some(rel.to_string_lossy().replace('\\', "/"));
    }
    if let Ok(canon_project) = project.canonicalize()
        && let Ok(rel) = path.strip_prefix(&canon_project)
    {
        return Some(rel.to_string_lossy().replace('\\', "/"));
    }
    None
}

/// 场景注册中心：在一个集合中维护多个场景，便于按名称检索。
pub struct ScenarioRegistry {
    scenarios: Vec<Box<dyn Scenario>>,
}

impl ScenarioRegistry {
    /// 创建空注册中心。
    pub fn new() -> Self {
        Self { scenarios: Vec::new() }
    }

    /// 创建包含开箱即用场景（文档 / 通用文本）的注册中心。
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.scenarios.push(Box::new(DocumentScenario::markdown()));
        reg.scenarios.push(Box::new(GenericTextScenario));
        reg
    }

    /// 注册一个场景。
    pub fn register(&mut self, scenario: Box<dyn Scenario>) {
        self.scenarios.push(scenario);
    }

    /// 按名称获取场景。
    pub fn get(&self, name: &str) -> Option<&dyn Scenario> {
        self.scenarios.iter().find(|s| s.name() == name).map(|s| s.as_ref())
    }

    /// 列出已注册的场景名称。
    pub fn names(&self) -> Vec<&str> {
        self.scenarios.iter().map(|s| s.name()).collect()
    }
}

impl Default for ScenarioRegistry {
    fn default() -> Self {
        Self::with_defaults()
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

    #[test]
    fn document_scenario_builds_tools() {
        let s = DocumentScenario::markdown();
        let tools = s.build_tools();
        assert_eq!(tools.len(), 3);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"read_text_file".to_string()));
        assert!(names.contains(&"list_docs".to_string()));
        assert!(names.contains(&"search_text".to_string()));
    }

    #[test]
    fn document_scenario_has_modifier() {
        let s = DocumentScenario::markdown();
        assert!(s.system_prompt_modifier().is_some());
    }

    #[test]
    fn generic_text_scenario_only_two_tools() {
        let s = GenericTextScenario;
        assert_eq!(s.build_tools().len(), 2);
    }

    #[test]
    fn read_text_file_returns_content() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("doc.md"), "# Title\nbody").unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("doc.md".into()));
        let call = ToolCall { name: "read_text_file".into(), arguments: args, id: None };
        let r = ReadTextFileTool.execute(&call, &ctx).unwrap();
        assert!(r.success);
        assert!(r.value["content"].as_str().unwrap().contains("Title"));
        assert_eq!(r.value["lines"].as_u64().unwrap(), 2);
    }

    #[test]
    fn list_docs_filters_extensions() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.md"), "x").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "y").unwrap();
        std::fs::write(tmp.path().join("ignore.rs"), "z").unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String(".".into()));
        let call = ToolCall { name: "list_docs".into(), arguments: args, id: None };
        let r = ListDocsTool { extensions: vec!["md".into(), "txt".into()] }
            .execute(&call, &ctx)
            .unwrap();
        assert!(r.success);
        let files: Vec<String> = r.value["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(files.contains(&"a.md".to_string()));
        assert!(files.contains(&"b.txt".to_string()));
        assert!(!files.iter().any(|f| f.ends_with(".rs")));
    }

    #[test]
    fn search_text_finds_matches() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.md"), "hello world\nfoo bar\nhello again").unwrap();
        let ctx = make_ctx(tmp.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("needle".to_string(), Value::String("hello".into()));
        let call = ToolCall { name: "search_text".into(), arguments: args, id: None };
        let r = SearchTextTool.execute(&call, &ctx).unwrap();
        assert!(r.success);
        let matches = r.value["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0]["line"].as_u64().unwrap(), 1);
    }

    #[test]
    fn scenario_registry_defaults() {
        let reg = ScenarioRegistry::with_defaults();
        let names = reg.names();
        assert!(names.contains(&"documents"));
        assert!(names.contains(&"generic-text"));
        assert!(reg.get("documents").is_some());
        assert!(reg.get("unknown").is_none());
    }

    #[test]
    fn scenario_registry_default_impl_uses_with_defaults() {
        let reg = ScenarioRegistry::default();
        assert!(!reg.names().is_empty());
    }
}