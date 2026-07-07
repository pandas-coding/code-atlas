//! 执行轨迹可视化与性能基准。
//!
//! 提供将 [`AgentResponse`] 渲染为 HTML / Markdown 报告的能力，以及
//! 基于历史响应的聚合性能基准统计。
//!
//! - [`TraceRenderer`]：将单次 Agent 执行的步骤、工具调用、指标渲染为
//!   HTML 或 Markdown 报告，便于归档与分享。
//! - [`PerformanceBenchmark`]：聚合多次 Agent 执行的响应，计算平均步数、
//!   耗时、token 效率等基准指标。

use std::collections::HashMap;

use crate::types::{AgentResponse, AgentState};

/// 执行轨迹渲染器：将 [`AgentResponse`] 转为 HTML 或 Markdown 报告。
pub struct TraceRenderer;

impl TraceRenderer {
    /// 渲染为 HTML 报告。
    ///
    /// 报告包含：摘要统计、步骤时间线、工具调用详情、自我评估（若有）。
    #[must_use]
    pub fn render_html(response: &AgentResponse) -> String {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str("<title>ChatVCode Agent Trace</title>\n");
        html.push_str("<style>\n");
        html.push_str(Self::html_css());
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str(&Self::render_html_header(response));
        html.push_str(&Self::render_html_summary(response));
        html.push_str(&Self::render_html_steps(response));
        if response.self_evaluation.is_some() {
            html.push_str(&Self::render_html_evaluation(response));
        }
        html.push_str("</body>\n</html>\n");
        html
    }

    /// 渲染为 Markdown 报告。
    ///
    /// 报告结构同 HTML 版本，但使用 Markdown 语法。
    #[must_use]
    pub fn render_markdown(response: &AgentResponse) -> String {
        let mut md = String::new();
        md.push_str("# ChatVCode Agent Execution Trace\n\n");
        md.push_str(&Self::render_md_summary(response));
        md.push_str(&Self::render_md_steps(response));
        if response.self_evaluation.is_some() {
            md.push_str(&Self::render_md_evaluation(response));
        }
        md
    }

    fn html_css() -> &'static str {
        r#"body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2em; color: #333; }
.summary { background: #f5f5f5; padding: 1em; border-radius: 8px; margin-bottom: 2em; }
.summary h2 { margin-top: 0; }
.stats { display: flex; flex-wrap: wrap; gap: 1em; }
.stat { background: white; padding: 0.8em 1.2em; border-radius: 6px; border: 1px solid #ddd; }
.stat .label { color: #888; font-size: 0.85em; text-transform: uppercase; }
.stat .value { font-size: 1.4em; font-weight: bold; color: #2563eb; }
.timeline { margin-top: 1em; }
.step { border-left: 3px solid #2563eb; padding-left: 1em; margin-bottom: 1.5em; }
.step-thinking { border-color: #6366f1; }
.step-acting { border-color: #10b981; }
.step-done { border-color: #6b7280; }
.step-failed { border-color: #ef4444; }
.step h3 { margin: 0 0 0.5em 0; }
.tool-call { background: #ecfdf5; padding: 0.5em 1em; border-radius: 4px; margin: 0.3em 0; font-family: monospace; font-size: 0.9em; }
.tool-result { background: #f0fdf4; padding: 0.5em 1em; border-radius: 4px; margin: 0.3em 0; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; word-break: break-all; }
.tool-error { background: #fef2f2; }
.thought { background: #eef2ff; padding: 0.5em 1em; border-radius: 4px; margin: 0.3em 0; font-style: italic; }
.evaluation { background: #fffbeb; padding: 1em; border-radius: 8px; margin-top: 2em; border: 1px solid #fde68a; }
.evaluation h2 { margin-top: 0; }
.confidence { font-size: 1.5em; font-weight: bold; }
.note { color: #92400e; }
.answer { background: #dbeafe; padding: 1em; border-radius: 8px; margin-top: 2em; }
"#
    }

    fn render_html_header(response: &AgentResponse) -> String {
        format!("<h1>ChatVCode Agent Execution Trace</h1>\n<p>Stop reason: <code>{:?}</code></p>\n", response.stop_reason)
    }

    fn render_html_summary(response: &AgentResponse) -> String {
        let mut html = String::new();
        html.push_str("<div class=\"summary\">\n<h2>Summary</h2>\n<div class=\"stats\">\n");
        html.push_str(&Self::html_stat("Steps", &response.metrics.total_steps.to_string()));
        html.push_str(&Self::html_stat("Tool calls", &response.total_tool_calls.to_string()));
        html.push_str(&Self::html_stat(
            "Tokens",
            &format!("{}", response.total_token_usage.total_tokens),
        ));
        html.push_str(&Self::html_stat(
            "Duration",
            &format!("{:.2}s", response.total_duration_ms as f64 / 1000.0),
        ));
        html.push_str(&Self::html_stat(
            "Success rate",
            &format!("{:.0}%", response.metrics.tool_success_rate * 100.0),
        ));
        html.push_str("</div>\n");
        if !response.answer.is_empty() {
            html.push_str(&format!(
                "<div class=\"answer\"><h3>Final Answer</h3><p>{}</p></div>\n",
                html_escape(&response.answer)
            ));
        }
        html.push_str("</div>\n");
        html
    }

    fn html_stat(label: &str, value: &str) -> String {
        format!(
            "<div class=\"stat\"><div class=\"label\">{label}</div><div class=\"value\">{value}</div></div>\n"
        )
    }

    fn render_html_steps(response: &AgentResponse) -> String {
        let mut html = String::new();
        html.push_str("<div class=\"timeline\">\n<h2>Execution Timeline</h2>\n");
        for step in &response.steps {
            let state_class = match step.state {
                AgentState::Thinking => "step-thinking",
                AgentState::Acting => "step-acting",
                AgentState::Done => "step-done",
                AgentState::Failed => "step-failed",
            };
            html.push_str(&format!(
                "<div class=\"step {state_class}\">\n<h3>Step {}: {:?}",
                step.step_number, step.state
            ));
            if let Some(phase) = step.thinking_phase {
                html.push_str(&format!(" ({:?})", phase));
            }
            html.push_str(&format!(
                " <small>{}ms · {} tokens</small></h3>\n",
                step.duration_ms, step.token_usage.total_tokens
            ));

            if let Some(thought) = &step.thought {
                html.push_str(&format!(
                    "<div class=\"thought\">{}</div>\n",
                    html_escape(thought)
                ));
            }

            for (idx, call) in step.tool_calls.iter().enumerate() {
                let result = step.tool_results.get(idx);
                html.push_str(&format!(
                    "<div class=\"tool-call\">→ {}({})</div>\n",
                    html_escape(&call.name),
                    call.arguments.len()
                ));
                if let Some(r) = result {
                    let class = if r.success { "tool-result" } else { "tool-result tool-error" };
                    let body = serde_json::to_string_pretty(&r.value).unwrap_or_default();
                    html.push_str(&format!(
                        "<div class=\"{class}\">{}</div>\n",
                        html_escape(&body)
                    ));
                }
            }
            html.push_str("</div>\n");
        }
        html.push_str("</div>\n");
        html
    }

    fn render_html_evaluation(response: &AgentResponse) -> String {
        let eval = response.self_evaluation.as_ref().unwrap();
        let mut html = String::new();
        html.push_str("<div class=\"evaluation\">\n<h2>Self-Evaluation</h2>\n");
        html.push_str(&format!(
            "<p class=\"confidence\">Confidence: {:.0}%</p>\n",
            eval.confidence_score * 100.0
        ));
        html.push_str("<ul>\n");
        html.push_str(&format!(
            "<li>Source citations: {}</li>\n",
            if eval.has_source_citations { "Yes" } else { "No" }
        ));
        html.push_str(&format!("<li>Answer length: {} chars</li>\n", eval.answer_length));
        html.push_str(&format!(
            "<li>Sufficient exploration: {}</li>\n",
            if eval.sufficient_exploration { "Yes" } else { "No" }
        ));
        html.push_str(&format!(
            "<li>Tool success rate: {:.0}%</li>\n",
            eval.tool_success_rate * 100.0
        ));
        html.push_str("</ul>\n");
        html.push_str("<h4>Notes:</h4>\n<ul>\n");
        for note in &eval.notes {
            html.push_str(&format!("<li class=\"note\">{}</li>\n", html_escape(note)));
        }
        html.push_str("</ul>\n</div>\n");
        html
    }

    fn render_md_summary(response: &AgentResponse) -> String {
        let mut md = String::new();
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n|--------|-------|\n");
        md.push_str(&format!("| Steps | {} |\n", response.metrics.total_steps));
        md.push_str(&format!("| Tool calls | {} |\n", response.total_tool_calls));
        md.push_str(&format!(
            "| Tokens | {} prompt + {} completion = {} total |\n",
            response.total_token_usage.prompt_tokens,
            response.total_token_usage.completion_tokens,
            response.total_token_usage.total_tokens
        ));
        md.push_str(&format!(
            "| Duration | {:.2}s |\n",
            response.total_duration_ms as f64 / 1000.0
        ));
        md.push_str(&format!(
            "| Tool success rate | {:.0}% |\n",
            response.metrics.tool_success_rate * 100.0
        ));
        md.push_str(&format!("| Stop reason | {:?} |\n\n", response.stop_reason));
        if !response.answer.is_empty() {
            md.push_str("### Final Answer\n\n");
            md.push_str(&response.answer);
            md.push_str("\n\n");
        }
        md
    }

    fn render_md_steps(response: &AgentResponse) -> String {
        let mut md = String::new();
        md.push_str("## Execution Timeline\n\n");
        for step in &response.steps {
            md.push_str(&format!(
                "### Step {} — {:?}",
                step.step_number, step.state
            ));
            if let Some(phase) = step.thinking_phase {
                md.push_str(&format!(" ({:?})", phase));
            }
            md.push_str(&format!(
                " _{}ms · {} tokens_\n\n",
                step.duration_ms, step.token_usage.total_tokens
            ));

            if let Some(thought) = &step.thought {
                md.push_str("> ");
                md.push_str(thought);
                md.push_str("\n\n");
            }

            for (idx, call) in step.tool_calls.iter().enumerate() {
                let result = step.tool_results.get(idx);
                md.push_str(&format!(
                    "- **{}**({} args)\n",
                    call.name,
                    call.arguments.len()
                ));
                if let Some(r) = result {
                    let status = if r.success { "✓" } else { "✗" };
                    md.push_str(&format!("  - {} Result: `{}`\n", status, truncate_str_md(&r.value.to_string(), 200)));
                }
            }
            md.push('\n');
        }
        md
    }

    fn render_md_evaluation(response: &AgentResponse) -> String {
        let eval = response.self_evaluation.as_ref().unwrap();
        let mut md = String::new();
        md.push_str("## Self-Evaluation\n\n");
        md.push_str(&format!("**Confidence: {:.0}%**\n\n", eval.confidence_score * 100.0));
        md.push_str("| Criterion | Value |\n|-----------|-------|\n");
        md.push_str(&format!(
            "| Source citations | {} |\n",
            if eval.has_source_citations { "Yes" } else { "No" }
        ));
        md.push_str(&format!("| Answer length | {} chars |\n", eval.answer_length));
        md.push_str(&format!(
            "| Sufficient exploration | {} |\n",
            if eval.sufficient_exploration { "Yes" } else { "No" }
        ));
        md.push_str(&format!(
            "| Tool success rate | {:.0}% |\n\n",
            eval.tool_success_rate * 100.0
        ));
        md.push_str("**Notes:**\n");
        for note in &eval.notes {
            md.push_str(&format!("- {}\n", note));
        }
        md
    }
}

/// 性能基准：聚合多次 Agent 执行，计算平均指标。
#[derive(Debug, Clone, Default)]
pub struct PerformanceBenchmark {
    /// 已记录的执行响应数量。
    pub run_count: usize,
    /// 平均步数。
    pub avg_steps: f64,
    /// 平均耗时（毫秒）。
    pub avg_duration_ms: f64,
    /// 平均 token 使用量。
    pub avg_tokens: f64,
    /// 平均工具调用数。
    pub avg_tool_calls: f64,
    /// 平均工具成功率。
    pub avg_tool_success_rate: f64,
    /// Token 效率（token / 步数）。
    pub token_efficiency: f64,
    /// 各工具的调用次数统计。
    pub tool_usage: HashMap<String, usize>,
    /// 各停止原因的出现次数。
    pub stop_reasons: HashMap<String, usize>,
}

impl PerformanceBenchmark {
    /// 创建空的基准统计器。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一次执行响应，更新聚合指标。
    pub fn add(&mut self, response: &AgentResponse) {
        self.run_count += 1;
        let n = self.run_count as f64;

        self.avg_steps = update_avg(self.avg_steps, response.metrics.total_steps as f64, n);
        self.avg_duration_ms = update_avg(self.avg_duration_ms, response.total_duration_ms as f64, n);
        self.avg_tokens = update_avg(self.avg_tokens, response.total_token_usage.total_tokens as f64, n);
        self.avg_tool_calls = update_avg(self.avg_tool_calls, response.total_tool_calls as f64, n);
        self.avg_tool_success_rate =
            update_avg(self.avg_tool_success_rate, response.metrics.tool_success_rate, n);

        self.token_efficiency = if self.avg_steps > 0.0 {
            self.avg_tokens / self.avg_steps
        } else {
            0.0
        };

        for (name, count) in &response.metrics.tool_calls_by_name {
            *self.tool_usage.entry(name.clone()).or_insert(0) += count;
        }

        let reason_str = format!("{:?}", response.stop_reason);
        *self.stop_reasons.entry(reason_str).or_insert(0) += 1;
    }

    /// 渲染为 Markdown 报告。
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# ChatVCode Agent Performance Benchmark\n\n");
        md.push_str(&format!("**Runs analyzed:** {}\n\n", self.run_count));
        md.push_str("| Metric | Average |\n|--------|---------|\n");
        md.push_str(&format!("| Steps | {:.2} |\n", self.avg_steps));
        md.push_str(&format!("| Duration | {:.2}ms ({:.2}s) |\n", self.avg_duration_ms, self.avg_duration_ms / 1000.0));
        md.push_str(&format!("| Tokens | {:.1} |\n", self.avg_tokens));
        md.push_str(&format!("| Tool calls | {:.2} |\n", self.avg_tool_calls));
        md.push_str(&format!("| Tool success rate | {:.1}% |\n", self.avg_tool_success_rate * 100.0));
        md.push_str(&format!("| Token efficiency | {:.1} tokens/step |\n\n", self.token_efficiency));

        if !self.tool_usage.is_empty() {
            md.push_str("## Tool Usage\n\n");
            md.push_str("| Tool | Calls |\n|------|-------|\n");
            let mut entries: Vec<_> = self.tool_usage.iter().collect();
            entries.sort_by(|a, b| b.1.cmp(a.1));
            for (name, count) in entries {
                md.push_str(&format!("| {} | {} |\n", name, count));
            }
            md.push('\n');
        }

        if !self.stop_reasons.is_empty() {
            md.push_str("## Stop Reasons\n\n");
            md.push_str("| Reason | Count |\n|--------|-------|\n");
            for (reason, count) in &self.stop_reasons {
                md.push_str(&format!("| {} | {} |\n", reason, count));
            }
        }

        md
    }
}

/// 增量平均更新：`old_avg * (n-1)/n + new_value/n`。
fn update_avg(old_avg: f64, new_value: f64, n: f64) -> f64 {
    if n <= 1.0 {
        new_value
    } else {
        old_avg * (n - 1.0) / n + new_value / n
    }
}

/// HTML 转义：&, <, >, ", '。
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// 截断字符串用于 Markdown 行内显示。
fn truncate_str_md(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let prefix: String = chars.into_iter().take(max).collect();
    format!("{prefix}...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentMetrics, AgentStep, AgentStopReason, AgentState, SelfEvaluation, SourceReference, ThinkingPhase, TokenUsage};
    use chatvcode_llm::{ToolCall, ToolResult};
    use serde_json::json;
    use std::collections::HashMap;

    fn make_response(answer: &str, steps: usize, duration_ms: u64, tokens: usize) -> AgentResponse {
        let mut metrics = AgentMetrics::default();
        metrics.total_steps = steps;
        metrics.tool_success_rate = 1.0;
        AgentResponse {
            answer: answer.into(),
            sources: vec![SourceReference {
                file_path: "src/main.rs".into(),
                line_start: 1,
                line_end: 10,
                symbol_name: Some("main".into()),
                relevance: 1.0,
            }],
            steps: (0..steps)
                .map(|i| AgentStep {
                    step_number: i + 1,
                    state: AgentState::Thinking,
                    thinking_phase: Some(ThinkingPhase::Planning),
                    thought: Some(format!("Thinking step {}", i + 1)),
                    tool_calls: vec![],
                    tool_results: vec![],
                    duration_ms: 100,
                    token_usage: TokenUsage {
                        prompt_tokens: tokens / 2,
                        completion_tokens: tokens / 2,
                        total_tokens: tokens,
                    },
                })
                .collect(),
            total_token_usage: TokenUsage {
                prompt_tokens: tokens / 2,
                completion_tokens: tokens / 2,
                total_tokens: tokens,
            },
            total_duration_ms: duration_ms,
            total_tool_calls: 0,
            stop_reason: AgentStopReason::Completed,
            metrics,
            self_evaluation: None,
        }
    }

    fn make_response_with_tool_call() -> AgentResponse {
        let mut metrics = AgentMetrics::default();
        metrics.total_steps = 2;
        metrics.tool_success_rate = 1.0;
        metrics.tool_calls_by_name.insert("read_file".into(), 1);

        let thinking = AgentStep {
            step_number: 1,
            state: AgentState::Thinking,
            thinking_phase: Some(ThinkingPhase::Planning),
            thought: Some("I need to read the file".into()),
            tool_calls: vec![],
            tool_results: vec![],
            duration_ms: 50,
            token_usage: TokenUsage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        };

        let acting = AgentStep {
            step_number: 2,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![ToolCall {
                name: "read_file".into(),
                arguments: HashMap::from([("path".to_string(), json!("main.rs"))]),
                id: None,
            }],
            tool_results: vec![ToolResult::success(json!({"content": "fn main() {}"}))],
            duration_ms: 100,
            token_usage: TokenUsage { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
        };

        AgentResponse {
            answer: "The main function is at src/main.rs:1".into(),
            sources: vec![],
            steps: vec![thinking, acting],
            total_token_usage: TokenUsage { prompt_tokens: 30, completion_tokens: 15, total_tokens: 45 },
            total_duration_ms: 150,
            total_tool_calls: 1,
            stop_reason: AgentStopReason::Completed,
            metrics,
            self_evaluation: None,
        }
    }

    #[test]
    fn render_html_contains_required_sections() {
        let resp = make_response_with_tool_call();
        let html = TraceRenderer::render_html(&resp);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Execution Trace"));
        assert!(html.contains("Summary"));
        assert!(html.contains("read_file"));
        assert!(html.contains("src/main.rs"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn render_markdown_contains_required_sections() {
        let resp = make_response_with_tool_call();
        let md = TraceRenderer::render_markdown(&resp);
        assert!(md.contains("# ChatVCode Agent Execution Trace"));
        assert!(md.contains("## Summary"));
        assert!(md.contains("read_file"));
        assert!(md.contains("Final Answer"));
    }

    #[test]
    fn render_html_escapes_special_characters() {
        let mut resp = make_response("answer", 1, 100, 50);
        resp.answer = "<script>alert('xss')</script>".into();
        let html = TraceRenderer::render_html(&resp);
        assert!(!html.contains("<script>alert('xss')</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn render_html_with_self_evaluation() {
        let mut resp = make_response("test answer that is long enough", 2, 200, 100);
        resp.self_evaluation = Some(SelfEvaluation {
            confidence_score: 0.75,
            has_source_citations: true,
            answer_length: 100,
            sufficient_exploration: true,
            tool_success_rate: 1.0,
            notes: vec!["Looks good".into()],
        });
        let html = TraceRenderer::render_html(&resp);
        assert!(html.contains("Self-Evaluation"));
        assert!(html.contains("75%"));
        assert!(html.contains("Looks good"));
    }

    #[test]
    fn render_markdown_with_self_evaluation() {
        let mut resp = make_response("test answer", 2, 200, 100);
        resp.self_evaluation = Some(SelfEvaluation {
            confidence_score: 0.5,
            has_source_citations: false,
            answer_length: 11,
            sufficient_exploration: true,
            tool_success_rate: 0.5,
            notes: vec!["Could be better".into()],
        });
        let md = TraceRenderer::render_markdown(&resp);
        assert!(md.contains("Self-Evaluation"));
        assert!(md.contains("50%"));
        assert!(md.contains("Could be better"));
    }

    #[test]
    fn benchmark_single_run() {
        let mut bench = PerformanceBenchmark::new();
        let resp = make_response("answer", 5, 1000, 500);
        bench.add(&resp);
        assert_eq!(bench.run_count, 1);
        assert_eq!(bench.avg_steps, 5.0);
        assert_eq!(bench.avg_duration_ms, 1000.0);
        assert_eq!(bench.avg_tokens, 500.0);
    }

    #[test]
    fn benchmark_multiple_runs_averages() {
        let mut bench = PerformanceBenchmark::new();
        bench.add(&make_response("a", 2, 100, 50));
        bench.add(&make_response("b", 4, 300, 150));
        assert_eq!(bench.run_count, 2);
        assert_eq!(bench.avg_steps, 3.0); // (2+4)/2
        assert_eq!(bench.avg_duration_ms, 200.0); // (100+300)/2
        assert_eq!(bench.avg_tokens, 100.0); // (50+150)/2
    }

    #[test]
    fn benchmark_tracks_tool_usage() {
        let mut bench = PerformanceBenchmark::new();
        bench.add(&make_response_with_tool_call());
        assert_eq!(bench.tool_usage.get("read_file"), Some(&1));
    }

    #[test]
    fn benchmark_tracks_stop_reasons() {
        let mut bench = PerformanceBenchmark::new();
        bench.add(&make_response("a", 1, 100, 50));
        assert!(bench.stop_reasons.contains_key("Completed"));
    }

    #[test]
    fn benchmark_render_markdown() {
        let mut bench = PerformanceBenchmark::new();
        bench.add(&make_response_with_tool_call());
        let md = bench.render_markdown();
        assert!(md.contains("Performance Benchmark"));
        assert!(md.contains("Runs analyzed"));
        assert!(md.contains("Tool Usage"));
        assert!(md.contains("read_file"));
    }

    #[test]
    fn html_escape_covers_all_special_chars() {
        assert_eq!(html_escape("a&b<c>d\"e'f"), "a&amp;b&lt;c&gt;d&quot;e&#x27;f");
    }

    #[test]
    fn truncate_str_md_short_and_long() {
        assert_eq!(truncate_str_md("hi", 10), "hi");
        let long = "x".repeat(50);
        let t = truncate_str_md(&long, 10);
        assert!(t.ends_with("..."));
    }
}
