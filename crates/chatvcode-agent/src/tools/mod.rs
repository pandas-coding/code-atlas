mod compare_files;
mod edit_file;
mod file_structure;
mod find_references;
mod get_dependencies;
mod get_project_overview;
mod grep_code;
mod list_files;
mod read_file;
mod run_command;
mod search_code;
mod search_symbol;
mod write_file;

pub use compare_files::CompareFilesTool;
pub use edit_file::EditFileTool;
pub use file_structure::GetFileStructureTool;
pub use find_references::FindReferencesTool;
pub use get_dependencies::GetDependenciesTool;
pub use get_project_overview::GetProjectOverviewTool;
pub use grep_code::GrepCodeTool;
pub use list_files::ListFilesTool;
pub use read_file::ReadFileTool;
pub use run_command::RunCommandTool;
pub use search_code::SearchCodeTool;
pub use search_symbol::SearchSymbolTool;
pub use write_file::WriteFileTool;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolResult};
use serde_json::Value;

use crate::context::ToolContext;
use crate::error::AgentError;

pub trait BuiltinTool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError>;

    fn validate_arguments(&self, call: &ToolCall) -> Result<(), AgentError> {
        let def = self.definition();
        for param in &def.parameters {
            if param.required && !call.arguments.contains_key(&param.name) {
                return Err(AgentError::ToolError {
                    tool_name: def.name.clone(),
                    message: format!("Missing required parameter: {}", param.name),
                });
            }
        }
        Ok(())
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            let msg = match &result.value {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            return format!("Error: {}", msg);
        }
        let text = match &result.value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if text.len() > 500 { format!("{}... (truncated)", &text[..500]) } else { text }
    }

    fn is_cacheable(&self) -> bool {
        true
    }

    fn cache_key(&self, call: &ToolCall) -> String {
        let mut keys: Vec<&String> = call.arguments.keys().collect();
        keys.sort();
        let args_str: Vec<String> = keys
            .iter()
            .map(|k| format!("{}={}", k, call.arguments[*k]))
            .collect();
        format!("{}:{}", call.name, args_str.join(","))
    }
}

pub fn register_all_tools() -> Vec<Box<dyn BuiltinTool>> {
    vec![
        Box::new(ReadFileTool),
        Box::new(ListFilesTool),
        Box::new(GrepCodeTool),
        Box::new(GetFileStructureTool),
        Box::new(SearchSymbolTool),
        Box::new(SearchCodeTool),
        Box::new(FindReferencesTool),
        Box::new(GetDependenciesTool),
        Box::new(CompareFilesTool),
        Box::new(GetProjectOverviewTool),
        Box::new(WriteFileTool),
        Box::new(EditFileTool),
        Box::new(RunCommandTool::new()),
    ]
}

pub fn build_tool_definitions(tools: &[Box<dyn BuiltinTool>]) -> Vec<ToolDefinition> {
    tools.iter().map(|t| t.definition()).collect()
}

pub fn find_tool<'a>(tools: &'a [Box<dyn BuiltinTool>], name: &str) -> Option<&'a dyn BuiltinTool> {
    tools
        .iter()
        .find(|t| t.definition().name == name)
        .map(|t| t.as_ref())
}

pub(crate) fn resolve_safe_path(
    project_path: &std::path::Path,
    file_path: &str,
) -> Result<std::path::PathBuf, AgentError> {
    let target = if std::path::Path::new(file_path).is_absolute() {
        std::path::PathBuf::from(file_path)
    } else {
        project_path.join(file_path)
    };
    let canonical_project = project_path
        .canonicalize()
        .unwrap_or_else(|_| project_path.to_path_buf());
    let canonical_target = target.canonicalize().unwrap_or_else(|_| {
        if let Some(parent) = target.parent() {
            if let Ok(cp) = parent.canonicalize() {
                if let Some(name) = target.file_name() {
                    return cp.join(name);
                }
            }
        }
        target.clone()
    });

    if !canonical_target.starts_with(&canonical_project) {
        return Err(AgentError::ToolError {
            tool_name: "path_check".into(),
            message: format!("Path '{}' is outside the project directory", file_path),
        });
    }
    Ok(canonical_target)
}

/// 写安全路径解析：在父目录尚未存在时也能正确解析路径（用于写入类工具）。
///
/// 与 [`resolve_safe_path`] 不同，本函数不依赖目标已存在：它规范化项目
/// 目录后，把相对 `file_path` 拼接到规范化项目下，再用纯词汇方式消除
/// `..`/`.`，最后做边界检查。
pub(crate) fn resolve_safe_path_for_write(
    project_path: &std::path::Path,
    file_path: &str,
) -> Result<std::path::PathBuf, AgentError> {
    use std::path::{Component, Path, PathBuf};

    let canonical_project = project_path
        .canonicalize()
        .map_err(|e| AgentError::ToolError {
            tool_name: "path_check".into(),
            message: format!("Cannot canonicalize project path '{}': {}", project_path.display(), e),
        })?;

    let joined: PathBuf = if Path::new(file_path).is_absolute() {
        PathBuf::from(file_path)
    } else {
        canonical_project.join(file_path)
    };

    // 词汇规范化：消除 ``.`` 与 ``..``，保留前缀（如 Windows ``\\?\``）
    let mut normalized = PathBuf::new();
    for comp in joined.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // 仅在仍有可弹出的非前缀组件时弹出
                let can_pop = normalized.parent().is_some()
                    && !matches!(
                        normalized.components().next_back(),
                        Some(Component::Prefix(_) | Component::RootDir)
                    );
                if can_pop {
                    normalized.pop();
                }
            }
            c => normalized.push(c.as_os_str()),
        }
    }

    if !normalized.starts_with(&canonical_project) {
        return Err(AgentError::ToolError {
            tool_name: "path_check".into(),
            message: format!("Path '{}' is outside the project directory", file_path),
        });
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_register_all_tools() {
        let tools = register_all_tools();
        assert_eq!(tools.len(), 13);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"list_files".to_string()));
        assert!(names.contains(&"grep_code".to_string()));
        assert!(names.contains(&"get_file_structure".to_string()));
        assert!(names.contains(&"search_symbol".to_string()));
        assert!(names.contains(&"search_code".to_string()));
        assert!(names.contains(&"find_references".to_string()));
        assert!(names.contains(&"get_dependencies".to_string()));
        assert!(names.contains(&"compare_files".to_string()));
        assert!(names.contains(&"get_project_overview".to_string()));
        assert!(names.contains(&"write_file".to_string()));
        assert!(names.contains(&"edit_file".to_string()));
        assert!(names.contains(&"run_command".to_string()));
    }

    #[test]
    fn test_build_tool_definitions() {
        let tools = register_all_tools();
        let defs = build_tool_definitions(&tools);
        assert_eq!(defs.len(), 13);
        for def in &defs {
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
        }
    }

    #[test]
    fn test_find_tool() {
        let tools = register_all_tools();
        assert!(find_tool(&tools, "read_file").is_some());
        assert!(find_tool(&tools, "nonexistent").is_none());
    }

    #[test]
    fn test_cache_key_deterministic() {
        let tool = ReadFileTool;
        let mut args1 = HashMap::new();
        args1.insert("path".to_string(), Value::String("a.rs".into()));
        args1.insert("offset".to_string(), Value::Number(1.into()));
        let call1 = ToolCall { name: "read_file".into(), arguments: args1, id: None };

        let mut args2 = HashMap::new();
        args2.insert("offset".to_string(), Value::Number(1.into()));
        args2.insert("path".to_string(), Value::String("a.rs".into()));
        let call2 = ToolCall { name: "read_file".into(), arguments: args2, id: None };

        assert_eq!(tool.cache_key(&call1), tool.cache_key(&call2));
    }

    #[test]
    fn test_summarize_result_default() {
        let tool = ReadFileTool;
        let ok = ToolResult::success(Value::String("hello".into()));
        assert_eq!(tool.summarize_result(&ok), "hello");

        let err = ToolResult::error("bad");
        assert_eq!(tool.summarize_result(&err), "Error: bad");
    }

    #[test]
    fn test_summarize_result_truncation() {
        let tool = ReadFileTool;
        let long = "x".repeat(600);
        let ok = ToolResult::success(Value::String(long));
        let summary = tool.summarize_result(&ok);
        assert!(summary.len() < 600);
        assert!(summary.ends_with("... (truncated)"));
    }

    #[test]
    fn test_validate_arguments_missing_required() {
        let tool = ReadFileTool;
        let call = ToolCall { name: "read_file".into(), arguments: HashMap::new(), id: None };
        assert!(tool.validate_arguments(&call).is_err());
    }

    #[test]
    fn test_validate_arguments_ok() {
        let tool = ReadFileTool;
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("test.rs".into()));
        let call = ToolCall { name: "read_file".into(), arguments: args, id: None };
        assert!(tool.validate_arguments(&call).is_ok());
    }
}
