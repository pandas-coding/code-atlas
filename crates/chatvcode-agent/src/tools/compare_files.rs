use std::fs;

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};

use crate::context::ToolContext;
use crate::error::AgentError;

use super::{BuiltinTool, resolve_safe_path};

/// `compare_files` 工具：逐行比较两个文本文件的差异。
///
/// 基于最长公共子序列（LCS）的简化实现，返回新增/删除/修改的行块。
/// 适用于 Agent 在重构前后对比文件、理解变更范围。
pub struct CompareFilesTool;

impl BuiltinTool for CompareFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("compare_files")
            .description(
                "Compare two text files line by line and return the differences. \
                 Useful for understanding changes between file versions or comparing \
                 similar files. Returns added, removed, and unchanged line blocks.",
            )
            .parameter(
                ToolParameter::string("path_a")
                    .description("Path to the first file (relative to project root)")
                    .required(true),
            )
            .parameter(
                ToolParameter::string("path_b")
                    .description("Path to the second file (relative to project root)")
                    .required(true),
            )
            .parameter(
                ToolParameter::integer("context_lines")
                    .description("Number of unchanged context lines to show around each diff block (default: 3)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let path_a = call.get_string("path_a").unwrap();
        let path_b = call.get_string("path_b").unwrap();
        let context_lines = call.get_i64("context_lines").unwrap_or(3).max(0) as usize;

        let resolved_a = resolve_safe_path(&ctx.project_path, path_a)?;
        let resolved_b = resolve_safe_path(&ctx.project_path, path_b)?;

        if !resolved_a.is_file() {
            return Ok(ToolResult::error(format!("Not a file: {}", path_a)));
        }
        if !resolved_b.is_file() {
            return Ok(ToolResult::error(format!("Not a file: {}", path_b)));
        }

        let content_a = fs::read_to_string(&resolved_a).map_err(|e| AgentError::ToolError {
            tool_name: "compare_files".into(),
            message: format!("Failed to read '{}': {}", path_a, e),
        })?;
        let content_b = fs::read_to_string(&resolved_b).map_err(|e| AgentError::ToolError {
            tool_name: "compare_files".into(),
            message: format!("Failed to read '{}': {}", path_b, e),
        })?;

        let lines_a: Vec<&str> = content_a.lines().collect();
        let lines_b: Vec<&str> = content_b.lines().collect();

        let diff = compute_diff(&lines_a, &lines_b, context_lines);

        let result = serde_json::json!({
            "path_a": path_a,
            "path_b": path_b,
            "lines_a": lines_a.len(),
            "lines_b": lines_b.len(),
            "blocks_added": diff.blocks_added,
            "blocks_removed": diff.blocks_removed,
            "blocks_unchanged": diff.blocks_unchanged,
            "identical": lines_a == lines_b,
            "diff": diff.blocks,
        });

        Ok(ToolResult::success(result))
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            return format!("Error: {}", result.value);
        }

        if result.value["identical"].as_bool().unwrap_or(false) {
            return "Files are identical".to_string();
        }

        let added = result.value["blocks_added"].as_u64().unwrap_or(0);
        let removed = result.value["blocks_removed"].as_u64().unwrap_or(0);
        format!("Files differ: {} added block(s), {} removed block(s)", added, removed)
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

/// 一个 diff 块：连续的同类型行（added / removed / context）。
#[derive(Debug, Clone, serde::Serialize)]
struct DiffBlock {
    /// 块类型：`context`、`added`、`removed`。
    kind: &'static str,
    /// 在文件 A 中的起始行号（1-indexed，removed/context 使用，added 为 0）。
    line_a_start: usize,
    /// 在文件 B 中的起始行号（1-indexed，added/context 使用，removed 为 0）。
    line_b_start: usize,
    /// 块内行数。
    count: usize,
    /// 块内的文本行。
    lines: Vec<String>,
}

/// diff 计算结果。
struct DiffResult {
    blocks: Vec<DiffBlock>,
    blocks_added: usize,
    blocks_removed: usize,
    blocks_unchanged: usize,
}

/// 基于最长公共子序列（LCS）的 diff 实现。
///
/// 通过动态规划求 LCS，再回溯生成 added/removed/context 块。
/// 对于大文件（>1000 行），LCS 表会占用 O(n*m) 内存，但工具场景下可接受。
fn compute_diff(lines_a: &[&str], lines_b: &[&str], context_lines: usize) -> DiffResult {
    let n = lines_a.len();
    let m = lines_b.len();

    // 构建 LCS 长度表
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if lines_a[i - 1] == lines_b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // 回溯生成操作序列
    #[derive(Debug, Clone, Copy)]
    enum Op {
        Equal,
        Added,
        Removed,
    }

    let mut ops: Vec<Op> = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && lines_a[i - 1] == lines_b[j - 1] {
            ops.push(Op::Equal);
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push(Op::Added);
            j -= 1;
        } else {
            ops.push(Op::Removed);
            i -= 1;
        }
    }
    ops.reverse();

    // 将操作序列合并为连续块
    // 注意：回溯得到的是按行号顺序的操作，但 Equal 行需要映射回 A/B 的行号
    let mut raw_blocks: Vec<DiffBlock> = Vec::new();
    let mut ia = 0usize; // A 的当前行索引
    let mut ib = 0usize; // B 的当前行索引

    for op in ops {
        let kind = match op {
            Op::Equal => "context",
            Op::Added => "added",
            Op::Removed => "removed",
        };
        let (line_a, line_b, line_content) = match op {
            Op::Equal => {
                let l = lines_a[ia].to_string();
                let a = ia + 1;
                let b = ib + 1;
                ia += 1;
                ib += 1;
                (a, b, l)
            }
            Op::Added => {
                let l = lines_b[ib].to_string();
                let b = ib + 1;
                ib += 1;
                (0, b, l)
            }
            Op::Removed => {
                let l = lines_a[ia].to_string();
                let a = ia + 1;
                ia += 1;
                (a, 0, l)
            }
        };

        // 与上一块同类且连续则合并
        if let Some(last) = raw_blocks.last_mut() {
            if last.kind == kind {
                last.count += 1;
                last.lines.push(line_content);
                continue;
            }
        }

        raw_blocks.push(DiffBlock {
            kind,
            line_a_start: line_a,
            line_b_start: line_b,
            count: 1,
            lines: vec![line_content],
        });
    }

    // 应用 context_lines：只保留 added/removed 块及其周围 context_lines 行的 context 块
    let blocks = if context_lines == 0 {
        // 只保留 added/removed 块
        raw_blocks.into_iter().filter(|b| b.kind != "context").collect()
    } else {
        apply_context_filter(raw_blocks, context_lines)
    };

    let mut blocks_added = 0;
    let mut blocks_removed = 0;
    let mut blocks_unchanged = 0;
    for b in &blocks {
        match b.kind {
            "added" => blocks_added += 1,
            "removed" => blocks_removed += 1,
            _ => blocks_unchanged += 1,
        }
    }

    DiffResult {
        blocks,
        blocks_added,
        blocks_removed,
        blocks_unchanged,
    }
}

/// 过滤 diff 块：仅保留 added/removed 块前后 `context_lines` 行的 context。
fn apply_context_filter(blocks: Vec<DiffBlock>, context_lines: usize) -> Vec<DiffBlock> {
    // 标记哪些块需要保留
    let n = blocks.len();
    let mut keep = vec![false; n];

    for (idx, b) in blocks.iter().enumerate() {
        if b.kind != "context" {
            // 保留自身及前后 context_lines 个 context 块
            keep[idx] = true;
            for offset in 1..=context_lines {
                if idx >= offset {
                    keep[idx - offset] = true;
                }
                if idx + offset < n {
                    keep[idx + offset] = true;
                }
            }
        }
    }

    // 提取保留的块，并对连续 context 块做截断
    let mut result: Vec<DiffBlock> = Vec::new();
    for (idx, b) in blocks.into_iter().enumerate() {
        if !keep[idx] {
            continue;
        }
        if b.kind == "context" && b.count > context_lines {
            // 截断过长的 context 块：保留前 context_lines 和后 context_lines 行
            // 简化处理：只保留前 context_lines 行（足够 Agent 理解上下文）
            let truncated_lines: Vec<String> =
                b.lines.iter().take(context_lines).cloned().collect();
            result.push(DiffBlock {
                kind: b.kind,
                line_a_start: b.line_a_start,
                line_b_start: b.line_b_start,
                count: truncated_lines.len(),
                lines: truncated_lines,
            });
        } else {
            result.push(b);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreTrait, CodeSearchService};
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use serde_json::Value;
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
    fn test_compare_identical_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "line1\nline2\nline3\n").unwrap();
        fs::write(dir.path().join("b.txt"), "line1\nline2\nline3\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = CompareFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path_a".to_string(), Value::String("a.txt".into()));
        args.insert("path_b".to_string(), Value::String("b.txt".into()));
        let call = ToolCall { name: "compare_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert!(result.value["identical"].as_bool().unwrap());
        assert_eq!(result.value["blocks_added"].as_u64().unwrap(), 0);
        assert_eq!(result.value["blocks_removed"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_compare_with_additions() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "line1\nline2\n").unwrap();
        fs::write(dir.path().join("b.txt"), "line1\nline2\nline3\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = CompareFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path_a".to_string(), Value::String("a.txt".into()));
        args.insert("path_b".to_string(), Value::String("b.txt".into()));
        let call = ToolCall { name: "compare_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert!(!result.value["identical"].as_bool().unwrap());
        assert!(result.value["blocks_added"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn test_compare_with_removals() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "line1\nline2\nline3\n").unwrap();
        fs::write(dir.path().join("b.txt"), "line1\nline3\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = CompareFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path_a".to_string(), Value::String("a.txt".into()));
        args.insert("path_b".to_string(), Value::String("b.txt".into()));
        let call = ToolCall { name: "compare_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert!(result.value["blocks_removed"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn test_compare_missing_file() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "content").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = CompareFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path_a".to_string(), Value::String("a.txt".into()));
        args.insert("path_b".to_string(), Value::String("missing.txt".into()));
        let call = ToolCall { name: "compare_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_compare_missing_params() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = CompareFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path_a".to_string(), Value::String("a.txt".into()));
        let call = ToolCall { name: "compare_files".into(), arguments: args, id: None };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_compare_definition() {
        let tool = CompareFilesTool;
        let def = tool.definition();
        assert_eq!(def.name, "compare_files");
        assert_eq!(def.required_params(), vec!["path_a", "path_b"]);
    }

    #[test]
    fn test_compare_summarize_identical() {
        let tool = CompareFilesTool;
        let result = ToolResult::success(serde_json::json!({
            "identical": true,
            "blocks_added": 0,
            "blocks_removed": 0,
        }));
        assert_eq!(tool.summarize_result(&result), "Files are identical");
    }
}
