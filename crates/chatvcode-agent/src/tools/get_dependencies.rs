use std::fs;
use std::io::{BufRead, BufReader};

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use regex::Regex;
use serde_json::Value;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::{BuiltinTool, resolve_safe_path};

/// `get_dependencies` 工具：分析源文件的依赖（import / require / use 语句）。
///
/// 通过针对各语言的正则模式提取 import/use/require 语句，返回该文件
/// 所依赖的模块/包列表。支持 Rust、TypeScript/JavaScript、Python、Go、Java。
pub struct GetDependenciesTool;

impl BuiltinTool for GetDependenciesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("get_dependencies")
            .description(
                "Analyze a source file and extract its dependencies (imports, requires, \
                 use statements). Supports Rust, TypeScript/JavaScript, Python, Go, and Java. \
                 Returns a list of dependency modules/packages with their import lines.",
            )
            .parameter(
                ToolParameter::string("path")
                    .description("Path to the source file to analyze (relative to project root)")
                    .required(true),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let path = call.get_string("path").unwrap();
        let resolved = resolve_safe_path(&ctx.project_path, path)?;

        if !resolved.is_file() {
            return Ok(ToolResult::error(format!("Not a file: {}", path)));
        }

        let source_text = fs::read_to_string(&resolved).map_err(|e| AgentError::ToolError {
            tool_name: "get_dependencies".into(),
            message: format!("Failed to read file '{}': {}", path, e),
        })?;

        let language = detect_language(&resolved);
        let patterns = dependency_patterns_for(language);

        let file = match fs::File::open(&resolved) {
            Ok(f) => f,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to open file '{}': {}",
                    path, e
                )))
            }
        };
        let reader = BufReader::new(file);

        let mut deps: Vec<Value> = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            for pattern in &patterns {
                let regex = match Regex::new(&pattern.regex) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                if let Some(captures) = regex.captures(&line) {
                    let module = captures
                        .get(1)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default();
                    if !module.is_empty() {
                        deps.push(serde_json::json!({
                            "module": module,
                            "kind": pattern.kind,
                            "line": line_num + 1,
                            "statement": line.trim(),
                        }));
                    }
                }
            }
        }

        let _ = source_text; // 已通过 reader 逐行读取

        let result = serde_json::json!({
            "path": path,
            "language": language,
            "dependency_count": deps.len(),
            "dependencies": deps,
        });

        Ok(ToolResult::success(result))
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            return format!("Error: {}", result.value);
        }

        let count = result.value["dependency_count"].as_u64().unwrap_or(0);
        if count == 0 {
            return "No dependencies found".to_string();
        }

        let lang = result.value["language"].as_str().unwrap_or("unknown");
        format!("Found {} dependencies ({})", count, lang)
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

/// 依赖匹配模式。
struct DependencyPattern {
    /// 正则表达式，第一个捕获组为模块/包名。
    regex: &'static str,
    /// 依赖类型（如 `use`、`import`、`require`）。
    kind: &'static str,
}

/// 根据文件扩展名推断语言。
fn detect_language(path: &std::path::Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "kt" => "kotlin",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" | "hh" => "cpp",
        _ => "other",
    }
}

/// 返回指定语言的依赖匹配模式列表。
fn dependency_patterns_for(language: &str) -> Vec<DependencyPattern> {
    match language {
        "rust" => vec![
            DependencyPattern {
                regex: r"^\s*use\s+([a-zA-Z0-9_:]+)",
                kind: "use",
            },
            DependencyPattern {
                regex: r#"^\s*extern\s+crate\s+([a-zA-Z0-9_]+)"#,
                kind: "extern_crate",
            },
        ],
        "typescript" | "javascript" => vec![
            // ES6 import
            DependencyPattern {
                regex: r#"^\s*import\s+.*from\s+['"]([^'"]+)['"]"#,
                kind: "import",
            },
            // Side-effect import
            DependencyPattern {
                regex: r#"^\s*import\s+['"]([^'"]+)['"]"#,
                kind: "import",
            },
            // require
            DependencyPattern {
                regex: r#"require\s*\(\s*['"]([^'"]+)['"]\s*\)"#,
                kind: "require",
            },
            // dynamic import
            DependencyPattern {
                regex: r#"import\s*\(\s*['"]([^'"]+)['"]\s*\)"#,
                kind: "dynamic_import",
            },
        ],
        "python" => vec![
            DependencyPattern {
                regex: r"^\s*import\s+([a-zA-Z0-9_.]+)",
                kind: "import",
            },
            DependencyPattern {
                regex: r"^\s*from\s+([a-zA-Z0-9_.]+)\s+import",
                kind: "from_import",
            },
        ],
        "go" => vec![DependencyPattern {
            regex: r#""([^"]+)""#,
            kind: "import",
        }],
        "java" | "kotlin" => vec![DependencyPattern {
            regex: r"^\s*import\s+([a-zA-Z0-9_.]+)",
            kind: "import",
        }],
        "c" | "cpp" => vec![DependencyPattern {
            regex: r#"^\s*#\s*include\s*[<"]([^>"]+)[>"]"#,
            kind: "include",
        }],
        _ => vec![],
    }
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
    fn test_get_dependencies_rust() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("main.rs"),
            "use std::io;\nuse std::collections::HashMap;\nfn main() {}\n",
        )
        .unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetDependenciesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("main.rs".into()));
        let call = ToolCall { name: "get_dependencies".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["language"].as_str().unwrap(), "rust");
        assert_eq!(result.value["dependency_count"].as_u64().unwrap(), 2);
        let deps = result.value["dependencies"].as_array().unwrap();
        assert_eq!(deps[0]["module"].as_str().unwrap(), "std::io");
        assert_eq!(deps[0]["kind"].as_str().unwrap(), "use");
    }

    #[test]
    fn test_get_dependencies_typescript() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("app.ts"),
            "import { foo } from './foo';\nconst bar = require('bar');\n",
        )
        .unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetDependenciesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("app.ts".into()));
        let call = ToolCall { name: "get_dependencies".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["language"].as_str().unwrap(), "typescript");
        assert_eq!(result.value["dependency_count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_get_dependencies_python() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("main.py"),
            "import os\nfrom collections import defaultdict\n",
        )
        .unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetDependenciesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("main.py".into()));
        let call = ToolCall { name: "get_dependencies".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["language"].as_str().unwrap(), "python");
        assert_eq!(result.value["dependency_count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_get_dependencies_not_found() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetDependenciesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("nonexistent.rs".into()));
        let call = ToolCall { name: "get_dependencies".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_get_dependencies_missing_path() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetDependenciesTool;
        let call = ToolCall {
            name: "get_dependencies".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_get_dependencies_no_deps() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("plain.rs"), "fn main() {\n    println!(\"hi\");\n}\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetDependenciesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("plain.rs".into()));
        let call = ToolCall { name: "get_dependencies".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["dependency_count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_get_dependencies_definition() {
        let tool = GetDependenciesTool;
        let def = tool.definition();
        assert_eq!(def.name, "get_dependencies");
        assert_eq!(def.required_params(), vec!["path"]);
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(std::path::Path::new("a.rs")), "rust");
        assert_eq!(detect_language(std::path::Path::new("b.ts")), "typescript");
        assert_eq!(detect_language(std::path::Path::new("c.py")), "python");
        assert_eq!(detect_language(std::path::Path::new("d.go")), "go");
        assert_eq!(detect_language(std::path::Path::new("e.unknown")), "other");
    }
}
