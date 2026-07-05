//! Agent 会话管理。
//!
//! [`AgentSession`] 跟踪整个 Agent 执行过程：状态机当前状态、思考阶段、
//! 步骤历史、底层 [`ChatSession`] 对话、累计 token 用量与耗时、工具结果缓存，
//! 以及 [`AgentMetrics`] 可观测性指标。
//!
//! 设计要点：
//! - 会话持有自己的 [`TokenBudgetManager`]，使 `remaining_budget()` 等
//!   预算查询自包含（`AgentLoop` 也可通过 [`AgentSession::budget_manager`]
//!   共享同一实例）。
//! - `add_step` 是更新步骤历史与累计指标的唯一入口，避免外部直接修改
//!   `steps` 而绕过指标累加。
//! - 终态（`Done`/`Failed`）通过 [`AgentSession::set_stop_reason`] 记录停止
//!   原因，供 `to_response()` 输出。

use std::time::Instant;

use chatvcode_llm::ChatSession;
use serde_json::Value;

use crate::budget::{SessionContext, TokenBudgetManager};
use crate::cache::ToolResultCache;
use crate::types::{
    AgentConfig, AgentMetrics, AgentResponse, AgentState, AgentStep, AgentStopReason,
    SourceReference, ThinkingPhase, TokenUsage,
};

/// Agent 会话：聚合一次 Agent 执行所需的全部可变状态。
#[derive(Debug)]
pub struct AgentSession {
    /// 会话唯一标识。
    id: String,
    /// Agent 配置（步数/超时/预算等限制来源）。
    config: AgentConfig,
    /// 当前 Agent 状态。
    state: AgentState,
    /// 当前思考阶段（仅在 `Thinking` 状态下被解释使用）。
    thinking_phase: ThinkingPhase,
    /// 步骤历史。
    steps: Vec<AgentStep>,
    /// 底层对话会话（系统提示词 + 多轮消息）。
    chat_session: ChatSession,
    /// 累计 token 使用。
    total_token_usage: TokenUsage,
    /// 累计耗时（毫秒），由各步骤 `duration_ms` 累加。
    total_duration_ms: u64,
    /// 会话开始时间。
    started_at: Instant,
    /// 工具结果缓存。
    tool_cache: ToolResultCache,
    /// Token 预算管理器。
    budget_manager: TokenBudgetManager,
    /// Agent 可观测性指标。
    metrics: AgentMetrics,
    /// 终态停止原因（仅在进入 `Done`/`Failed` 后设置）。
    stop_reason: Option<AgentStopReason>,
    /// 最终回答文本（`Thinking -> Done` 时记录）。
    final_answer: Option<String>,
}

impl AgentSession {
    /// 创建一个新的 Agent 会话，初始状态 `Thinking` / `Planning`。
    ///
    /// 预算管理器与工具缓存分别基于 `config.token_budget` 与默认策略构建；
    /// 系统提示词由 `AgentLoop` 在运行前通过 [`ChatSession::set_system_prompt`]
    /// 注入，故此处不预先设置。
    #[must_use]
    pub fn new(config: AgentConfig) -> Self {
        let chat_template = config.chat_template.clone();
        let budget_manager = TokenBudgetManager::new(config.token_budget.clone());
        let chat_session = ChatSession::new(chat_template)
            .max_context_tokens(config.token_budget.total_budget)
            .reserve_for_response(config.token_budget.response_reserve);

        Self {
            id: generate_session_id(),
            config,
            state: AgentState::Thinking,
            thinking_phase: ThinkingPhase::Planning,
            steps: Vec::new(),
            chat_session,
            total_token_usage: TokenUsage::default(),
            total_duration_ms: 0,
            started_at: Instant::now(),
            tool_cache: ToolResultCache::default(),
            budget_manager,
            metrics: AgentMetrics::default(),
            stop_reason: None,
            final_answer: None,
        }
    }

    /// 当前步骤序号（即已记录的步骤数量）。
    ///
    /// 第一步执行前为 0；记录首步后为 1。
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.steps.len()
    }

    /// 记录一个新步骤，并同步更新累计 token、耗时与指标。
    ///
    /// 步骤的 `step_number` 会被规范化为 `current_step() + 1`，避免调用方
    /// 传错序号导致统计错乱。
    pub fn add_step(&mut self, mut step: AgentStep) {
        step.step_number = self.steps.len() + 1;
        self.total_token_usage.merge(&step.token_usage);
        self.total_duration_ms += step.duration_ms;
        self.metrics.merge_step(&step);
        self.steps.push(step);
        self.metrics.recompute_derived(&self.steps);
    }

    /// 是否处于终态 `Done`。
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.state == AgentState::Done
    }

    /// 是否处于终态 `Failed`。
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.state == AgentState::Failed
    }

    /// 是否已超时。
    ///
    /// `timeout_secs == 0` 视为不限制超时。
    #[must_use]
    pub fn is_timed_out(&self) -> bool {
        self.config.timeout_secs > 0 && self.elapsed_ms() >= self.config.timeout_secs * 1000
    }

    /// 剩余可用步数（`max_steps` 与已执行步数之差，下界 0）。
    #[must_use]
    pub fn remaining_steps(&self) -> usize {
        self.config.max_steps.saturating_sub(self.steps.len())
    }

    /// 剩余 token 预算（已扣除响应预留）。
    ///
    /// 基于会话内置的 [`TokenBudgetManager`] 与当前 `chat_session` 内容计算。
    #[must_use]
    pub fn remaining_budget(&self) -> usize {
        let system = self.chat_session.get_system_prompt().unwrap_or("");
        let messages = self.chat_session.messages();
        let ctx = SessionContext { system_prompt: system, messages };
        self.budget_manager.remaining_tokens(&ctx)
    }

    /// 已用 token 预算（系统提示词 + 历史）。
    #[must_use]
    pub fn used_budget(&self) -> usize {
        let system = self.chat_session.get_system_prompt().unwrap_or("");
        let messages = self.chat_session.messages();
        let ctx = SessionContext { system_prompt: system, messages };
        self.budget_manager.used_tokens(&ctx)
    }

    /// 将当前会话状态打包为 [`AgentResponse`]。
    ///
    /// - `answer`：优先取 `final_answer`；否则回退到最后一个 `Thinking` 步骤
    ///   的 `thought`；都没有则返回空字符串。
    /// - `sources`：尽力从步骤的工具结果中提取代码来源引用。
    /// - `stop_reason`：若未显式设置，则按当前状态推导（`Done` -> `Completed`，
    ///   `Failed` -> `Error`）。
    #[must_use]
    pub fn to_response(&self) -> AgentResponse {
        let answer = self.final_answer.clone().or_else(|| {
            self.steps
                .iter()
                .rev()
                .find(|s| s.state == AgentState::Thinking)
                .and_then(|s| s.thought.clone())
        }).unwrap_or_default();

        let sources = extract_sources(&self.steps);

        let total_tool_calls: usize = self.steps.iter().map(|s| s.tool_calls.len()).sum();

        let stop_reason = self.stop_reason.clone().unwrap_or_else(|| match self.state {
            AgentState::Done => AgentStopReason::Completed,
            AgentState::Failed => AgentStopReason::Error("agent failed".into()),
            AgentState::Thinking | AgentState::Acting => {
                if self.is_timed_out() {
                    AgentStopReason::Timeout
                } else if self.remaining_steps() == 0 {
                    AgentStopReason::MaxSteps
                } else {
                    AgentStopReason::Completed
                }
            }
        });

        AgentResponse {
            answer,
            sources,
            steps: self.steps.clone(),
            total_token_usage: self.total_token_usage.clone(),
            total_duration_ms: self.total_duration_ms,
            total_tool_calls,
            stop_reason,
            metrics: self.metrics.clone(),
        }
    }

    // ---- 访问器 -----------------------------------------------------------

    /// 会话 ID。
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Agent 配置引用。
    #[must_use]
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Agent 配置可变引用（用于 `continue_execution` 放宽步数限制等场景）。
    pub fn config_mut(&mut self) -> &mut AgentConfig {
        &mut self.config
    }

    /// 当前 Agent 状态。
    #[must_use]
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// 设置 Agent 状态（由 `AgentLoop` 在状态转移时调用）。
    pub fn set_state(&mut self, state: AgentState) {
        self.state = state;
    }

    /// 当前思考阶段。
    #[must_use]
    pub fn thinking_phase(&self) -> ThinkingPhase {
        self.thinking_phase
    }

    /// 设置思考阶段（仅在 `Thinking` 状态下有意义，但不在此处强制）。
    pub fn set_thinking_phase(&mut self, phase: ThinkingPhase) {
        self.thinking_phase = phase;
    }

    /// 步骤历史（只读视图）。
    #[must_use]
    pub fn steps(&self) -> &[AgentStep] {
        &self.steps
    }

    /// 底层对话会话（只读）。
    #[must_use]
    pub fn chat_session(&self) -> &ChatSession {
        &self.chat_session
    }

    /// 底层对话会话（可变）。
    pub fn chat_session_mut(&mut self) -> &mut ChatSession {
        &mut self.chat_session
    }

    /// 累计 token 使用。
    #[must_use]
    pub fn total_token_usage(&self) -> &TokenUsage {
        &self.total_token_usage
    }

    /// 累计耗时（毫秒）。
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.total_duration_ms
    }

    /// 自会话创建以来经过的毫秒数。
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    /// 工具结果缓存引用。
    #[must_use]
    pub fn tool_cache(&self) -> &ToolResultCache {
        &self.tool_cache
    }

    /// Token 预算管理器引用。
    #[must_use]
    pub fn budget_manager(&self) -> &TokenBudgetManager {
        &self.budget_manager
    }

    /// 可观测性指标（只读）。
    #[must_use]
    pub fn metrics(&self) -> &AgentMetrics {
        &self.metrics
    }

    /// 可观测性指标（可变，用于记录缓存命中/重试/循环检测等外部事件）。
    pub fn metrics_mut(&mut self) -> &mut AgentMetrics {
        &mut self.metrics
    }

    /// 终态停止原因。
    #[must_use]
    pub fn stop_reason(&self) -> Option<&AgentStopReason> {
        self.stop_reason.as_ref()
    }

    /// 设置停止原因（进入终态时调用）。
    pub fn set_stop_reason(&mut self, reason: AgentStopReason) {
        self.stop_reason = Some(reason);
    }

    /// 最终回答文本（只读）。
    #[must_use]
    pub fn final_answer(&self) -> Option<&str> {
        self.final_answer.as_deref()
    }

    /// 设置最终回答文本。
    pub fn set_final_answer(&mut self, answer: impl Into<String>) {
        self.final_answer = Some(answer.into());
    }

    // ---- 指标记录便捷方法 -------------------------------------------------

    /// 记录一次工具缓存命中。
    pub fn record_cache_hit(&mut self) {
        self.metrics.record_cache_hit();
    }

    /// 记录一次工具重试。
    pub fn record_retry(&mut self) {
        self.metrics.record_retry();
    }

    /// 记录一次循环检测触发。
    pub fn record_loop_detection(&mut self) {
        self.metrics.record_loop_detection();
    }
}

/// 从步骤的工具结果中尽力提取代码来源引用。
///
/// 支持以下结果结构（由内置工具产出）：
/// - `search_code` / `search_symbol`：`{ results: [{ file, start_line, end_line, symbol, score }] }`
/// - `grep_code`：`{ matches: [{ file, line }] }`
/// - `read_file`：`{ file, start_line, end_line }`
///
/// 未能识别的结构会被静默跳过；提取结果按相关性分数（若有）降序排序。
fn extract_sources(steps: &[AgentStep]) -> Vec<SourceReference> {
    let mut sources = Vec::new();
    for step in steps {
        for result in &step.tool_results {
            if !result.success {
                continue;
            }
            collect_source_from_value(&result.value, &mut sources);
        }
    }
    sources.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sources
}

/// 递归地从 JSON 值中提取来源引用。
fn collect_source_from_value(value: &serde_json::Value, out: &mut Vec<SourceReference>) {
    use serde_json::Value;

    match value {
        Value::Object(map) => {
            // 若该对象本身看起来像一个来源条目（含 file 字段），直接提取。
            if let Some(src) = source_from_object(map) {
                out.push(src);
            }
            // 同时递归数组字段（如 results / matches），以兼容聚合型结果。
            for key in ["results", "matches", "files", "items"].iter() {
                if let Some(Value::Array(arr)) = map.get(*key) {
                    for item in arr {
                        if let Value::Object(obj) = item
                            && let Some(src) = source_from_object(obj)
                        {
                            out.push(src);
                        }
                    }
                }
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_source_from_value(item, out);
            }
        }
        _ => {}
    }
}

/// 尝试将一个 JSON 对象转换为 [`SourceReference`]。
///
/// 必须包含 `file`（或 `file_path`）字段；行号缺省为 0，符号/分数可选。
fn source_from_object(map: &serde_json::Map<String, Value>) -> Option<SourceReference> {
    let file_path = map
        .get("file")
        .or_else(|| map.get("file_path"))
        .and_then(|v| v.as_str())?
        .to_string();

    let line_start = map
        .get("start_line")
        .or_else(|| map.get("line"))
        .and_then(|v| v.as_u64())
        .map_or(0, |n| n as usize);
    let line_end = map
        .get("end_line")
        .and_then(|v| v.as_u64())
        .map_or(line_start, |n| n as usize);

    let symbol_name = map
        .get("symbol")
        .or_else(|| map.get("symbol_name"))
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let relevance = map
        .get("score")
        .or_else(|| map.get("relevance"))
        .and_then(|v| v.as_f64())
        .map_or(0.0, |f| f as f32);

    Some(SourceReference {
        file_path,
        line_start,
        line_end: line_end.max(line_start),
        symbol_name,
        relevance,
    })
}

/// 生成一个简单的会话唯一标识。
///
/// 与 `chatvcode-llm` 的 `ChatSession` ID 风格保持一致，非完整 UUID。
fn generate_session_id() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("agent-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chatvcode_llm::{ToolCall, ToolResult};
    use serde_json::json;

    fn minimal_config() -> AgentConfig {
        AgentConfig {
            max_steps: 3,
            timeout_secs: 0,
            ..AgentConfig::default()
        }
    }

    fn step(state: AgentState, duration_ms: u64, usage: TokenUsage) -> AgentStep {
        AgentStep {
            step_number: 0,
            state,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![],
            tool_results: vec![],
            duration_ms,
            token_usage: usage,
        }
    }

    #[test]
    fn new_session_initial_state() {
        let s = AgentSession::new(minimal_config());
        assert_eq!(s.state(), AgentState::Thinking);
        assert_eq!(s.thinking_phase(), ThinkingPhase::Planning);
        assert_eq!(s.current_step(), 0);
        assert_eq!(s.remaining_steps(), 3);
        assert!(!s.is_done());
        assert!(!s.is_failed());
        assert!(s.stop_reason().is_none());
        assert!(s.final_answer().is_none());
        assert!(!s.id().is_empty());
    }

    #[test]
    fn add_step_updates_counters_and_normalizes_number() {
        let mut s = AgentSession::new(minimal_config());
        let mut st = step(AgentState::Thinking, 100, TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        });
        st.step_number = 99; // 故意错误
        s.add_step(st);

        assert_eq!(s.current_step(), 1);
        assert_eq!(s.steps()[0].step_number, 1); // 已规范化
        assert_eq!(s.total_duration_ms(), 100);
        assert_eq!(s.total_token_usage().total_tokens, 15);
        assert_eq!(s.metrics().total_steps, 1);
        assert_eq!(s.metrics().thinking_time_ms, 100);
        assert_eq!(s.remaining_steps(), 2);
    }

    #[test]
    fn add_step_accumulates_across_steps() {
        let mut s = AgentSession::new(minimal_config());
        s.add_step(step(AgentState::Thinking, 50, TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 0,
            total_tokens: 10,
        }));
        s.add_step(step(AgentState::Acting, 30, TokenUsage {
            prompt_tokens: 20,
            completion_tokens: 5,
            total_tokens: 25,
        }));

        assert_eq!(s.current_step(), 2);
        assert_eq!(s.total_duration_ms(), 80);
        assert_eq!(s.total_token_usage().prompt_tokens, 30);
        assert_eq!(s.total_token_usage().completion_tokens, 5);
        assert_eq!(s.total_token_usage().total_tokens, 35);
        assert_eq!(s.metrics().thinking_time_ms, 50);
        assert_eq!(s.metrics().acting_time_ms, 30);
    }

    #[test]
    fn is_timed_out_respects_zero_timeout() {
        let mut cfg = minimal_config();
        cfg.timeout_secs = 0;
        let s = AgentSession::new(cfg);
        assert!(!s.is_timed_out());
    }

    #[test]
    fn is_timed_out_triggers_after_expiry() {
        let mut cfg = minimal_config();
        cfg.timeout_secs = 1; // 1000ms
        let s = AgentSession::new(cfg);
        // 会话刚创建，不应超时
        assert!(!s.is_timed_out());
    }

    #[test]
    fn remaining_steps_saturates_at_zero() {
        let mut s = AgentSession::new(minimal_config()); // max_steps = 3
        for _ in 0..5 {
            s.add_step(step(AgentState::Thinking, 1, TokenUsage::default()));
        }
        assert_eq!(s.remaining_steps(), 0);
    }

    #[test]
    fn remaining_budget_decreases_with_messages() {
        let mut s = AgentSession::new(minimal_config());
        let initial = s.remaining_budget();
        s.chat_session_mut().set_system_prompt(Some("a".repeat(400)));
        let after = s.remaining_budget();
        assert!(after < initial, "{after} should be < {initial}");
    }

    #[test]
    fn to_response_uses_final_answer_when_set() {
        let mut s = AgentSession::new(minimal_config());
        s.set_final_answer("the answer");
        let resp = s.to_response();
        assert_eq!(resp.answer, "the answer");
        assert_eq!(resp.total_tool_calls, 0);
        assert!(resp.sources.is_empty());
    }

    #[test]
    fn to_response_falls_back_to_last_thought() {
        let mut s = AgentSession::new(minimal_config());
        let mut st = step(AgentState::Thinking, 10, TokenUsage::default());
        st.thought = Some("a conclusion".into());
        s.add_step(st);
        let resp = s.to_response();
        assert_eq!(resp.answer, "a conclusion");
    }

    #[test]
    fn to_response_stop_reason_derived_from_state() {
        let mut s = AgentSession::new(minimal_config());
        s.set_state(AgentState::Done);
        let resp = s.to_response();
        assert!(matches!(resp.stop_reason, AgentStopReason::Completed));

        s.set_state(AgentState::Failed);
        let resp = s.to_response();
        assert!(matches!(resp.stop_reason, AgentStopReason::Error(_)));
    }

    #[test]
    fn to_response_stop_reason_explicit_overrides_state() {
        let mut s = AgentSession::new(minimal_config());
        s.set_state(AgentState::Done);
        s.set_stop_reason(AgentStopReason::Timeout);
        let resp = s.to_response();
        assert!(matches!(resp.stop_reason, AgentStopReason::Timeout));
    }

    #[test]
    fn to_response_stop_reason_max_steps_when_exhausted() {
        let mut cfg = minimal_config();
        cfg.max_steps = 1;
        let mut s = AgentSession::new(cfg);
        s.add_step(step(AgentState::Thinking, 1, TokenUsage::default()));
        // 仍在 Thinking，但步数已用完
        let resp = s.to_response();
        assert!(matches!(resp.stop_reason, AgentStopReason::MaxSteps));
    }

    #[test]
    fn to_response_extracts_sources_from_search_results() {
        let mut s = AgentSession::new(minimal_config());
        let mut st = step(AgentState::Acting, 10, TokenUsage::default());
        st.tool_results = vec![ToolResult::success(json!({
            "results": [
                { "file": "src/a.rs", "start_line": 1, "end_line": 5, "symbol": "foo", "score": 0.9 },
                { "file": "src/b.rs", "start_line": 10, "symbol": "bar", "score": 0.5 },
            ]
        }))];
        s.add_step(st);

        let resp = s.to_response();
        assert_eq!(resp.sources.len(), 2);
        // 按分数降序
        assert_eq!(resp.sources[0].file_path, "src/a.rs");
        assert_eq!(resp.sources[0].line_start, 1);
        assert_eq!(resp.sources[0].line_end, 5);
        assert_eq!(resp.sources[0].symbol_name.as_deref(), Some("foo"));
        assert!(resp.sources[0].relevance > resp.sources[1].relevance);
    }

    #[test]
    fn to_response_extracts_sources_from_grep_matches() {
        let mut s = AgentSession::new(minimal_config());
        let mut st = step(AgentState::Acting, 10, TokenUsage::default());
        st.tool_results = vec![ToolResult::success(json!({
            "matches": [
                { "file": "src/c.rs", "line": 42, "text": "fn main" },
            ]
        }))];
        s.add_step(st);

        let resp = s.to_response();
        assert_eq!(resp.sources.len(), 1);
        assert_eq!(resp.sources[0].file_path, "src/c.rs");
        assert_eq!(resp.sources[0].line_start, 42);
    }

    #[test]
    fn to_response_skips_failed_tool_results() {
        let mut s = AgentSession::new(minimal_config());
        let mut st = step(AgentState::Acting, 10, TokenUsage::default());
        st.tool_results = vec![
            ToolResult::error("boom"),
            ToolResult::success(json!({ "results": [{ "file": "x.rs", "start_line": 1 }] })),
        ];
        s.add_step(st);

        let resp = s.to_response();
        assert_eq!(resp.sources.len(), 1);
        assert_eq!(resp.sources[0].file_path, "x.rs");
    }

    #[test]
    fn to_response_counts_total_tool_calls() {
        let mut s = AgentSession::new(minimal_config());
        let mut st1 = step(AgentState::Acting, 10, TokenUsage::default());
        st1.tool_calls = vec![
            ToolCall { name: "read_file".into(), arguments: Default::default(), id: None },
            ToolCall { name: "grep_code".into(), arguments: Default::default(), id: None },
        ];
        let mut st2 = step(AgentState::Acting, 10, TokenUsage::default());
        st2.tool_calls = vec![
            ToolCall { name: "read_file".into(), arguments: Default::default(), id: None },
        ];
        s.add_step(st1);
        s.add_step(st2);

        let resp = s.to_response();
        assert_eq!(resp.total_tool_calls, 3);
        assert_eq!(resp.metrics.tool_calls_by_name.get("read_file"), Some(&2));
        assert_eq!(resp.metrics.tool_calls_by_name.get("grep_code"), Some(&1));
    }

    #[test]
    fn record_methods_increment_metrics() {
        let mut s = AgentSession::new(minimal_config());
        s.record_cache_hit();
        s.record_cache_hit();
        s.record_retry();
        s.record_loop_detection();
        assert_eq!(s.metrics().tool_cache_hits, 2);
        assert_eq!(s.metrics().tool_retries, 1);
        assert_eq!(s.metrics().loop_detection_triggered, 1);
    }

    #[test]
    fn set_state_and_thinking_phase_roundtrip() {
        let mut s = AgentSession::new(minimal_config());
        s.set_state(AgentState::Acting);
        s.set_thinking_phase(ThinkingPhase::Observing);
        assert_eq!(s.state(), AgentState::Acting);
        assert_eq!(s.thinking_phase(), ThinkingPhase::Observing);
    }

    #[test]
    fn budget_manager_reflects_config() {
        let mut cfg = minimal_config();
        cfg.token_budget.total_budget = 5000;
        let s = AgentSession::new(cfg);
        assert_eq!(s.budget_manager().config().total_budget, 5000);
    }

    #[test]
    fn chat_session_uses_configured_template() {
        use chatvcode_llm::ChatTemplate;
        let mut cfg = minimal_config();
        cfg.chat_template = ChatTemplate::Llama3;
        let s = AgentSession::new(cfg);
        assert_eq!(s.chat_session().template(), &ChatTemplate::Llama3);
    }
}
