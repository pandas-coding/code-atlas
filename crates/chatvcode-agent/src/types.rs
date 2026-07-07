use std::collections::HashMap;
use std::path::PathBuf;

use chatvcode_llm::{ChatTemplate, GenerationParams, ToolCall, ToolResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Thinking,
    Acting,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingPhase {
    Planning,
    Observing,
    Concluding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStopReason {
    Completed,
    MaxSteps,
    Timeout,
    UserCancel,
    LoopDetected,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub step_number: usize,
    pub state: AgentState,
    pub thinking_phase: Option<ThinkingPhase>,
    pub thought: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub duration_ms: u64,
    pub token_usage: TokenUsage,
}

#[derive(Debug, Clone)]
pub struct TokenBudgetConfig {
    pub total_budget: usize,
    pub system_prompt_reserve: usize,
    pub tool_result_max: usize,
    pub history_budget: usize,
    pub response_reserve: usize,
}

impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            total_budget: 8192,
            system_prompt_reserve: 1500,
            tool_result_max: 2000,
            history_budget: 4000,
            response_reserve: 1500,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolRetryConfig {
    pub max_retries: usize,
    pub retry_on_timeout: bool,
    pub retry_on_transient_error: bool,
    pub backoff_ms: u64,
}

impl Default for ToolRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_on_timeout: true,
            retry_on_transient_error: true,
            backoff_ms: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub timeout_secs: u64,
    pub max_tool_calls_per_step: usize,
    pub allowed_tools: Vec<String>,
    pub project_path: PathBuf,
    pub verbose: bool,
    pub generation_params: GenerationParams,
    pub system_prompt: Option<String>,
    pub chat_template: ChatTemplate,
    pub token_budget: TokenBudgetConfig,
    pub tool_retry: ToolRetryConfig,
    pub require_plan_confirmation: bool,
    pub auto_approve_threshold: usize,
    /// 是否在同一步骤中并行执行多个工具调用（默认 false，顺序执行）。
    pub enable_parallel_tool_calls: bool,
    /// 是否在最终回答后执行自我评估（默认 false）。
    pub enable_self_evaluation: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            timeout_secs: 120,
            max_tool_calls_per_step: 5,
            allowed_tools: vec![],
            project_path: PathBuf::from("."),
            verbose: false,
            generation_params: GenerationParams::default(),
            system_prompt: None,
            chat_template: ChatTemplate::Auto,
            token_budget: TokenBudgetConfig::default(),
            tool_retry: ToolRetryConfig::default(),
            require_plan_confirmation: false,
            auto_approve_threshold: 3,
            enable_parallel_tool_calls: false,
            enable_self_evaluation: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub symbol_name: Option<String>,
    pub relevance: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub total_steps: usize,
    pub tool_calls_by_name: HashMap<String, usize>,
    pub tool_success_rate: f64,
    pub avg_tool_latency_ms: u64,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub thinking_time_ms: u64,
    pub acting_time_ms: u64,
    pub loop_detection_triggered: usize,
    pub tool_cache_hits: usize,
    pub tool_retries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub answer: String,
    pub sources: Vec<SourceReference>,
    pub steps: Vec<AgentStep>,
    pub total_token_usage: TokenUsage,
    pub total_duration_ms: u64,
    pub total_tool_calls: usize,
    pub stop_reason: AgentStopReason,
    pub metrics: AgentMetrics,
    /// Agent 自我评估结果（仅在 `enable_self_evaluation = true` 时填充）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_evaluation: Option<SelfEvaluation>,
}

/// Agent 对自身回答质量的启发式评估。
///
/// 不依赖额外 LLM 调用，而是基于回答长度、来源引用、工具成功率等
/// 指标计算一个 0.0-1.0 的置信度分数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvaluation {
    /// 综合置信度分数（0.0 - 1.0）。
    pub confidence_score: f64,
    /// 回答是否引用了至少一个代码来源。
    pub has_source_citations: bool,
    /// 回答字符长度。
    pub answer_length: usize,
    /// 是否达到最少步数（避免过早结束）。
    pub sufficient_exploration: bool,
    /// 工具调用成功率（0.0 - 1.0）。
    pub tool_success_rate: f64,
    /// 评估产生的建议（人类可读）。
    pub notes: Vec<String>,
}

impl SelfEvaluation {
    /// 基于响应指标计算自我评估。
    ///
    /// 评分维度：
    /// - 来源引用：有引用 +0.3
    /// - 回答长度：>200 字符 +0.2，>500 +0.1
    /// - 探索充分性：步数 >= 2 +0.2
    /// - 工具成功率：按比例 *0.2
    pub fn evaluate(response: &AgentResponse) -> Self {
        let has_source_citations = !response.sources.is_empty();
        let answer_length = response.answer.chars().count();
        let sufficient_exploration = response.steps.len() >= 2;
        let tool_success_rate = response.metrics.tool_success_rate;

        let mut score = 0.0f64;
        let mut notes = Vec::new();

        if has_source_citations {
            score += 0.3;
        } else {
            notes.push("Answer lacks code source citations (file:line references).".into());
        }

        if answer_length > 200 {
            score += 0.2;
            if answer_length > 500 {
                score += 0.1;
            }
        } else {
            notes.push("Answer is short; consider elaborating with more detail.".into());
        }

        if sufficient_exploration {
            score += 0.2;
        } else {
            notes.push("Limited exploration; answer may benefit from more tool calls.".into());
        }

        score += tool_success_rate * 0.2;

        if tool_success_rate < 0.5 && response.total_tool_calls > 0 {
            notes.push(format!(
                "Low tool success rate ({:.0}%); some explorations failed.",
                tool_success_rate * 100.0
            ));
        }

        if notes.is_empty() {
            notes.push("Answer appears well-supported by codebase exploration.".into());
        }

        SelfEvaluation {
            confidence_score: score.min(1.0),
            has_source_citations,
            answer_length,
            sufficient_exploration,
            tool_success_rate,
            notes,
        }
    }
}

impl AgentResponse {
    /// 计算并附加自我评估结果，返回新的响应。
    #[must_use]
    pub fn with_self_evaluation(mut self) -> Self {
        self.self_evaluation = Some(SelfEvaluation::evaluate(&self));
        self
    }
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    StateChanged { from: AgentState, to: AgentState },
    ThinkingPhaseChanged { phase: ThinkingPhase },
    Thinking { text: String },
    ToolCallStarted { name: String, arguments: serde_json::Value },
    ToolCallCompleted { name: String, result: ToolResult, cached: bool },
    ToolCallFailed { name: String, error: String, will_retry: bool },
    ToolCallRetrying { name: String, attempt: usize, max_attempts: usize },
    StepCompleted { step: AgentStep },
    LoopDetected { detection_type: serde_json::Value },
    AnswerStarted,
    AnswerToken { text: String },
    AnswerCompleted { response: AgentResponse },
    Error { message: String },
    TokenBudgetWarning { used: usize, remaining: usize },
}
