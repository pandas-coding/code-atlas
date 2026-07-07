use std::collections::HashMap;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use serde_json::Value;
use walkdir::WalkDir;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

/// `get_project_overview` 工具：生成项目结构概览。
///
/// 扫描项目目录，统计文件按语言/扩展名分布、目录结构、关键配置文件
/// （如 `Cargo.toml`、`package.json`、`go.mod`），帮助 Agent 快速建立
/// 对项目整体结构的认知。
pub struct GetProjectOverviewTool;

impl BuiltinTool for GetProjectOverviewTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("get_project_overview")
            .description(
                "Generate an overview of the project structure: file counts by language, \
                 top-level directories, and key manifest files (Cargo.toml, package.json, \
                 go.mod, etc.). Useful for quickly understanding a project's layout and \
                 tech stack.",
            )
            .parameter(
                ToolParameter::string("path")
                    .description("Directory to overview (relative to project root, default: \".\")"),
            )
            .parameter(
                ToolParameter::integer("max_depth")
                    .description("Maximum directory depth to traverse (default: 3)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let sub_path = call.get_string("path").unwrap_or(".");
        let max_depth = call.get_i64("max_depth").unwrap_or(3).max(1) as usize;

        let target_dir = if sub_path == "." {
            ctx.project_path.clone()
        } else {
            ctx.project_path.join(sub_path)
        };

        if !target_dir.is_dir() {
            return Ok(ToolResult::error(format!("Not a directory: {}", sub_path)));
        }

        let canonical_project = ctx
            .project_path
            .canonicalize()
            .unwrap_or_else(|_| ctx.project_path.clone());
        let canonical_target = target_dir
            .canonicalize()
            .unwrap_or_else(|_| target_dir.clone());
        if !canonical_target.starts_with(&canonical_project) {
            return Ok(ToolResult::error(format!(
                "Path '{}' is outside the project directory",
                sub_path
            )));
        }

        let mut total_files = 0usize;
        let mut total_size: u64 = 0;
        let mut by_extension: HashMap<String, usize> = HashMap::new();
        let mut by_language: HashMap<String, usize> = HashMap::new();
        let mut manifest_files: Vec<Value> = Vec::new();
        let mut directory_tree: Vec<String> = Vec::new();

        // 收集目录树（仅目录，限 max_depth）
        for entry in WalkDir::new(&canonical_target)
            .follow_links(true)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_dir() {
                let rel = entry
                    .path()
                    .strip_prefix(&canonical_target)
                    .ok()
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_default();
                if !rel.is_empty() {
                    let depth = rel.matches('/').count();
                    let indent = "  ".repeat(depth);
                    let dir_name = rel.rsplit('/').next().unwrap_or(&rel);
                    directory_tree.push(format!("{}{}/", indent, dir_name));
                }
            }
        }

        // 收集文件统计
        for entry in WalkDir::new(&canonical_target).follow_links(true).into_iter() {
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

            // 跳过常见忽略目录
            if is_ignored_path(&rel_path) {
                continue;
            }

            total_files += 1;
            let metadata = entry.metadata().ok();
            if let Some(m) = &metadata {
                total_size += m.len();
            }

            // 按扩展名统计
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_else(|| "(no ext)".into());
            let lang = extension_to_language(&ext);
            *by_extension.entry(ext).or_insert(0) += 1;

            // 按语言统计
            *by_language.entry(lang.to_string()).or_insert(0) += 1;

            // 顶层目录统计在单独的循环中完成（见下方 top_dir_counts）

            // 关键 manifest 文件
            let file_name = entry.file_name().to_string_lossy().to_string();
            if is_manifest_file(&file_name) {
                let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                manifest_files.push(serde_json::json!({
                    "file": rel_path,
                    "name": file_name,
                    "size": size,
                }));
            }
        }

        // 重新计算 top_dirs（顶层一级子目录的文件计数）
        let mut top_dir_counts: HashMap<String, usize> = HashMap::new();
        for entry in WalkDir::new(&canonical_target)
            .follow_links(true)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry
                .path()
                .strip_prefix(&canonical_target)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            if rel.is_empty() {
                continue;
            }
            if is_ignored_path(&rel) {
                continue;
            }
            let top = rel.split('/').next().unwrap_or(&rel);
            *top_dir_counts.entry(top.to_string()).or_insert(0) += 1;
        }

        let mut top_dirs_vec: Vec<(String, usize)> = top_dir_counts.into_iter().collect();
        top_dirs_vec.sort_by(|a, b| b.1.cmp(&a.1));

        let mut ext_vec: Vec<(String, usize)> = by_extension.into_iter().collect();
        ext_vec.sort_by(|a, b| b.1.cmp(&a.1));

        let mut lang_vec: Vec<(String, usize)> = by_language.into_iter().collect();
        lang_vec.sort_by(|a, b| b.1.cmp(&a.1));

        let result = serde_json::json!({
            "path": sub_path,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "by_extension": ext_vec.into_iter().map(|(k, v)| serde_json::json!({"extension": k, "count": v})).collect::<Vec<_>>(),
            "by_language": lang_vec.into_iter().map(|(k, v)| serde_json::json!({"language": k, "count": v})).collect::<Vec<_>>(),
            "top_directories": top_dirs_vec.into_iter().map(|(name, count)| serde_json::json!({"name": name, "file_count": count})).collect::<Vec<_>>(),
            "manifest_files": manifest_files,
            "directory_tree": directory_tree,
            "max_depth": max_depth,
        });

        Ok(ToolResult::success(result))
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            return format!("Error: {}", result.value);
        }

        let total = result.value["total_files"].as_u64().unwrap_or(0);
        let langs = result.value["by_language"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .take(3)
                    .filter_map(|l| {
                        let name = l["language"].as_str()?;
                        let count = l["count"].as_u64()?;
                        Some(format!("{}({})", name, count))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        format!("Project: {} files (top languages: {})", total, langs)
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

/// 判断路径是否属于应忽略的目录（如 `.git`、`node_modules`、`target`）。
fn is_ignored_path(rel_path: &str) -> bool {
    let ignored = [
        ".git", ".svn", ".hg", "node_modules", "target", "build", "dist", "__pycache__",
        ".venv", "venv", ".cache", ".idea", ".vscode",
    ];
    for part in rel_path.split('/') {
        if ignored.contains(&part) {
            return true;
        }
    }
    false
}

/// 判断文件名是否为关键 manifest 文件。
fn is_manifest_file(name: &str) -> bool {
    matches!(
        name,
        "Cargo.toml"
            | "package.json"
            | "go.mod"
            | "pyproject.toml"
            | "requirements.txt"
            | "pom.xml"
            | "build.gradle"
            | "build.gradle.kts"
            | "CMakeLists.txt"
            | "Makefile"
            | "Gemfile"
            | "mix.exs"
            | "composer.json"
    )
}

/// 文件扩展名到语言的映射。
fn extension_to_language(ext: &str) -> &'static str {
    match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" | "hh" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "scala" | "sc" => "scala",
        "sh" | "bash" => "shell",
        "sql" => "sql",
        "html" | "htm" => "html",
        "css" | "scss" | "sass" => "css",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "md" => "markdown",
        "xml" => "xml",
        _ => "other",
    }
}

// top_dirs 占位变量类型修正
#[allow(unused)]
fn _top_dirs_type_check() -> HashMap<String, usize> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreTrait, CodeSearchService};
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use std::fs;
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
    fn test_project_overview_basic() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::write(dir.path().join("b.rs"), "fn b() {}").unwrap();
        fs::write(dir.path().join("c.txt"), "hello").unwrap();
        fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetProjectOverviewTool;
        let call = ToolCall {
            name: "get_project_overview".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert!(result.value["total_files"].as_u64().unwrap() >= 4);
        let manifests = result.value["manifest_files"].as_array().unwrap();
        assert!(manifests.iter().any(|m| m["name"].as_str() == Some("Cargo.toml")));
    }

    #[test]
    fn test_project_overview_ignores_target() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::create_dir_all(dir.path().join("target/debug")).unwrap();
        fs::write(dir.path().join("target/debug/binary.o"), "binary").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetProjectOverviewTool;
        let call = ToolCall {
            name: "get_project_overview".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        // target 目录下的文件被忽略
        assert_eq!(result.value["total_files"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_project_overview_by_language() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::write(dir.path().join("b.rs"), "").unwrap();
        fs::write(dir.path().join("c.py"), "").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetProjectOverviewTool;
        let call = ToolCall {
            name: "get_project_overview".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };

        let result = tool.execute(&call, &ctx).unwrap();
        let langs = result.value["by_language"].as_array().unwrap();
        let rust_count = langs
            .iter()
            .find(|l| l["language"].as_str() == Some("rust"))
            .map(|l| l["count"].as_u64().unwrap())
            .unwrap_or(0);
        assert_eq!(rust_count, 2);
    }

    #[test]
    fn test_project_overview_not_a_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file.txt"), "content").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetProjectOverviewTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("file.txt".into()));
        let call = ToolCall { name: "get_project_overview".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_project_overview_definition() {
        let tool = GetProjectOverviewTool;
        let def = tool.definition();
        assert_eq!(def.name, "get_project_overview");
        assert!(def.required_params().is_empty());
    }

    #[test]
    fn test_is_manifest_file() {
        assert!(is_manifest_file("Cargo.toml"));
        assert!(is_manifest_file("package.json"));
        assert!(is_manifest_file("go.mod"));
        assert!(!is_manifest_file("main.rs"));
    }

    #[test]
    fn test_extension_to_language() {
        assert_eq!(extension_to_language("rs"), "rust");
        assert_eq!(extension_to_language("py"), "python");
        assert_eq!(extension_to_language("unknown"), "other");
    }
}
