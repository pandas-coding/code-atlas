//! Agent 自动 Prompt 优化：生成多个系统提示词变体，
//! 在同一查询上分别运行 Agent，依启发式指标比较变体并选出最佳。
//!
//! 该模块不强依赖真实 LLM——其内部使用 [`PromptRunRunner`] trait，
//! 调用方可注入真实 Agent 执行器（如包装
//! [`agent_query`](crate::service::agent_query)）或测试桩。
//!
//! 该实现遵循「P2 可延后项」的简化定位：核心提供了一个可测试的 A/B
//! 变体比较骨架，并未引入额外模型微调/网格搜索算法。

use crate::error::{AgentError, AgentResult};
use crate::types::AgentResponse;

/// 一条系统提示词变体。
#[derive(Debug, Clone)]
pub struct PromptVariant {
    /// 变体唯一标识。
    pub name: String,
    /// 变体的完整系统提示词文本。
    pub system_prompt: String,
}

impl PromptVariant {
    /// 创建一个变体。
    pub fn new(name: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        Self { name: name.into(), system_prompt: system_prompt.into() }
    }
}

/// 单条变体的运行指标汇总。
///
/// 指标取自 [`AgentResponse`]，便于在不具备真实 LLM 的环境下使用启发式评估。
#[derive(Debug, Clone, Default)]
pub struct PromptMetric {
    /// 系统提示词长度（字符）。
    pub prompt_length: usize,
    /// 回答字符长度。
    pub answer_length: usize,
    /// 工具调用成功率（0.0 - 1.0）。
    pub tool_success_rate: f64,
    /// 总步数。
    pub total_steps: usize,
    /// 总 token 使用量。
    pub total_tokens: usize,
    /// 总耗时（毫秒）。
    pub total_duration_ms: u64,
    /// 是否引用了来源。
    pub has_sources: bool,
}

impl PromptMetric {
    /// 从 [`AgentResponse`] 与变体收集指标。
    pub fn from_response(variant: &PromptVariant, response: &AgentResponse) -> Self {
        Self {
            prompt_length: variant.system_prompt.chars().count(),
            answer_length: response.answer.chars().count(),
            tool_success_rate: response.metrics.tool_success_rate,
            total_steps: response.steps.len(),
            total_tokens: response.total_token_usage.total_tokens,
            total_duration_ms: response.total_duration_ms,
            has_sources: !response.sources.is_empty(),
        }
    }
}

/// 变体评估结果。
#[derive(Debug, Clone)]
pub struct VariantScore {
    /// 变体名称。
    pub variant_name: String,
    /// 启发式分数（越高越好，无上界约束）。
    pub score: f64,
    /// 产生该分数的指标。
    pub metric: PromptMetric,
}

impl VariantScore {
    /// 评估函数：基于启发式权重计算分数。
    ///
    /// 权重说明：
    /// - 来源引用 +1.5
    /// - 工具成功率 * 1.5
    /// - 探索步数 min(steps/3, 1) * 1.0 （过度探索不再加分）
    /// - 回答长度 Normalize 处理：reasonable 在 200~2000 字符之间加分 0.5
    /// - Prompt 过长惩罚：超过 1000 字符每千字符 -0.2
    /// - Token 效率：answer_length/max(total_tokens,1)*0.5
    pub fn evaluate(variant: &PromptVariant, response: &AgentResponse) -> Self {
        let m = PromptMetric::from_response(variant, response);
        let mut score = 0.0f64;
        if m.has_sources {
            score += 1.5;
        }
        score += m.tool_success_rate.clamp(0.0, 1.0) * 1.5;
        score += (m.total_steps as f64 / 3.0).clamp(0.0, 1.0);
        if m.answer_length >= 200 && m.answer_length <= 2000 {
            score += 0.5;
        }
        // prompt 过长惩罚
        if m.prompt_length > 1000 {
            let excess = m.prompt_length - 1000;
            score -= 0.2 * (excess as f64 / 1000.0);
        }
        let efficiency = m.answer_length as f64 / (m.total_tokens.max(1) as f64);
        score += efficiency * 0.5;
        Self {
            variant_name: variant.name.clone(),
            score: score.max(0.0),
            metric: m,
        }
    }
}

/// 一次优化的最终结果。
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// 最佳变体的索引（在原始变体列表中）。
    pub best_variant_index: usize,
    /// 最佳变体名称。
    pub best_variant_name: String,
    /// 最佳变体分数。
    pub best_score: f64,
    /// 所有变体的分数（按输入顺序）。
    pub scores: Vec<VariantScore>,
}

/// 一次对单条变体的运行能力。
pub trait PromptRunRunner: Send + Sync {
    /// 在使用 `system_prompt`（覆盖默认 system prompt）时，执行指定查询并返回响应。
    fn run_with_prompt(&self, query: &str, system_prompt: &str) -> AgentResult<AgentResponse>;
}

/// 自动 Prompt 优化器：传入一组变体与运行器，选出最佳变体。
pub struct PromptOptimizer<'a> {
    variants: Vec<PromptVariant>,
    runner: &'a dyn PromptRunRunner,
}

impl<'a> PromptOptimizer<'a> {
    /// 创建一个优化器。
    pub fn new(variants: Vec<PromptVariant>, runner: &'a dyn PromptRunRunner) -> Self {
        Self { variants, runner }
    }

    /// 在指定查询上比较所有变体，返回最佳变体索引及其分数。
    ///
    /// 任一变体运行失败不算终止整体优化，仅记录该变体得分为 0。
    pub fn optimize(&self, query: &str) -> Result<OptimizationResult, AgentError> {
        if self.variants.is_empty() {
            return Err(AgentError::ConfigError("No prompt variants provided".into()));
        }

        let mut scores: Vec<VariantScore> = Vec::with_capacity(self.variants.len());

        for variant in &self.variants {
            let score = match self.runner.run_with_prompt(query, &variant.system_prompt) {
                Ok(response) => VariantScore::evaluate(variant, &response),
                Err(e) => {
                    log::warn!("Variant '{}' failed: {}", variant.name, e);
                    VariantScore {
                        variant_name: variant.name.clone(),
                        score: 0.0,
                        metric: PromptMetric {
                            prompt_length: variant.system_prompt.chars().count(),
                            answer_length: 0,
                            tool_success_rate: 0.0,
                            total_steps: 0,
                            total_tokens: 0,
                            total_duration_ms: 0,
                            has_sources: false,
                        },
                    }
                }
            };
            scores.push(score);
        }

        let (best_index, best) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.score
                    .partial_cmp(&b.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty scores");

        Ok(OptimizationResult {
            best_variant_index: best_index,
            best_variant_name: best.variant_name.clone(),
            best_score: best.score,
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AgentMetrics, AgentResponse, AgentStopReason, AgentStep, SourceReference, TokenUsage,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn synth(answer: &str, sources: bool, tool_rate: f64, steps: usize, tokens: usize) -> AgentResponse {
        let sources_list = if sources {
            vec![SourceReference {
                file_path: "a.rs".into(),
                line_start: 1,
                line_end: 5,
                symbol_name: Some("foo".into()),
                relevance: 0.9,
            }]
        } else {
            Vec::<SourceReference>::new()
        };
        AgentResponse {
            answer: answer.to_string(),
            sources: sources_list,
            steps: (0..steps)
                .map(|i| AgentStep {
                    step_number: i + 1,
                    state: crate::types::AgentState::Acting,
                    thinking_phase: None,
                    thought: None,
                    tool_calls: vec![],
                    tool_results: vec![],
                    duration_ms: 10,
                    token_usage: TokenUsage::default(),
                })
                .collect(),
            total_token_usage: TokenUsage {
                prompt_tokens: tokens / 2,
                completion_tokens: tokens / 2,
                total_tokens: tokens,
            },
            total_duration_ms: 100,
            total_tool_calls: steps,
            stop_reason: AgentStopReason::Completed,
            metrics: AgentMetrics {
                tool_success_rate: tool_rate,
                total_steps: steps,
                ..Default::default()
            },
            self_evaluation: None,
        }
    }

    struct CannedRunner {
        answer: String,
        calls: Arc<AtomicUsize>,
    }
    impl PromptRunRunner for CannedRunner {
        fn run_with_prompt(&self, _query: &str, _system_prompt: &str) -> AgentResult<AgentResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(synth(
                &self.answer,
                true,
                0.9,
                3,
                200,
            ))
        }
    }

    struct FailingRunner;
    impl PromptRunRunner for FailingRunner {
        fn run_with_prompt(&self, _: &str, _: &str) -> AgentResult<AgentResponse> {
            Err(AgentError::Internal("injected failure".into()))
        }
    }

    #[test]
    fn optimizer_picks_best_variant() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = CannedRunner { answer: "answer".into(), calls: Arc::clone(&calls) };
        let variants = vec![
            PromptVariant::new("a", "prompt A"),
            PromptVariant::new("b", "prompt B - longer than A and richer description, so scoring might differ"),
        ];
        let opt = PromptOptimizer::new(variants, &runner);
        let result = opt.optimize("test").unwrap();
        assert_eq!(result.scores.len(), 2);
        assert!(result.best_score > 0.0);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn optimizer_handles_empty_variants() {
        let runner = CannedRunner { answer: "x".into(), calls: Arc::new(AtomicUsize::new(0)) };
        let opt = PromptOptimizer::new(vec![], &runner);
        let result = opt.optimize("q");
        assert!(result.is_err());
    }

    #[test]
    fn optimizer_records_zero_score_on_failure() {
        let runner = FailingRunner;
        let variants = vec![PromptVariant::new("a", "p")];
        let opt = PromptOptimizer::new(variants, &runner);
        let result = opt.optimize("q").unwrap();
        assert_eq!(result.scores.len(), 1);
        assert_eq!(result.scores[0].score, 0.0);
        assert_eq!(result.best_variant_name, "a");
    }

    #[test]
    fn variant_score_evaluates_dimensions() {
        let v = PromptVariant::new("x", "short");
        let resp = synth("a thousand characters answer".to_string().as_str(), true, 1.0, 4, 100);
        let score = VariantScore::evaluate(&v, &resp);
        assert!(score.score > 0.0);
        assert!(score.metric.has_sources);
        assert_eq!(score.metric.total_steps, 4);
    }

    #[test]
    fn long_prompt_is_penalized() {
        let v_short = PromptVariant::new("s", "p");
        let v_long = PromptVariant::new("l", "x".repeat(2200));
        let resp = synth("answer", true, 0.8, 2, 100);
        let score_short = VariantScore::evaluate(&v_short, &resp).score;
        let score_long = VariantScore::evaluate(&v_long, &resp).score;
        assert!(score_long < score_short);
    }

    #[test]
    fn metric_from_response_extracts_fields() {
        let v = PromptVariant::new("v", "prompt".to_string());
        let resp = synth("hello world this is an answer of moderate length", false, 0.5, 1, 500);
        let m = PromptMetric::from_response(&v, &resp);
        assert_eq!(m.prompt_length, 6);
        assert!(!m.has_sources);
        assert_eq!(m.total_tokens, 500);
        assert_eq!(m.tool_success_rate, 0.5);
    }
}