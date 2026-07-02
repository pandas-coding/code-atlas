//! Agent 指标收集与累加逻辑。
//!
//! 提供 [`TokenUsage`] 的累加与 [`AgentMetrics`] 的步骤合并/派生指标计算，
//! 供 [`crate::session::AgentSession`] 在 `add_step` 时同步更新可观测性数据。

use crate::types::{AgentMetrics, AgentState, AgentStep, TokenUsage};

impl TokenUsage {
    /// 累加另一份 token 用量。
    ///
    /// `total_tokens` 会被重置为 `prompt_tokens + completion_tokens` 之和，
    /// 避免上游未正确设置 total 时累加错误。
    pub fn merge(&mut self, other: &TokenUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }

    /// 累加原始 prompt/completion token 数（不依赖 `other.total_tokens`）。
    pub fn add_usage(&mut self, prompt_tokens: usize, completion_tokens: usize) {
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }
}

impl AgentMetrics {
    /// 合并一个新步骤产生的原始计数（token、耗时、工具调用次数）。
    ///
    /// 派生指标（成功率、平均延迟）不会在此方法中更新，需额外调用
    /// [`AgentMetrics::recompute_derived`] 以基于全部步骤重新计算，
    /// 保证统计口径一致。
    pub fn merge_step(&mut self, step: &AgentStep) {
        // token 累计
        self.total_input_tokens += step.token_usage.prompt_tokens;
        self.total_output_tokens += step.token_usage.completion_tokens;

        // 耗时按状态归类
        match step.state {
            AgentState::Thinking => self.thinking_time_ms += step.duration_ms,
            AgentState::Acting => self.acting_time_ms += step.duration_ms,
            AgentState::Done | AgentState::Failed => {}
        }

        // 工具调用计数
        for call in &step.tool_calls {
            *self.tool_calls_by_name.entry(call.name.clone()).or_insert(0) += 1;
        }
    }

    /// 基于全部步骤重新计算派生指标（步数、成功率、平均工具延迟）。
    ///
    /// 在 [`crate::session::AgentSession::add_step`] 之后调用，确保
    /// `tool_success_rate` / `avg_tool_latency_ms` 与当前 `steps` 一致。
    pub fn recompute_derived(&mut self, steps: &[AgentStep]) {
        self.total_steps = steps.len();

        let mut total_results = 0usize;
        let mut success_results = 0usize;
        let mut latency_sum = 0u64;
        let mut latency_count = 0usize;

        for step in steps {
            for result in &step.tool_results {
                total_results += 1;
                if result.success {
                    success_results += 1;
                }
            }

            // 仅 Acting 步骤的耗时用于工具延迟统计；该步骤内多个工具结果
            // 共享同一个 duration_ms，按结果数量均摊以得到平均单次延迟。
            if step.state == AgentState::Acting && !step.tool_results.is_empty() {
                latency_sum += step.duration_ms;
                latency_count += step.tool_results.len();
            }
        }

        self.tool_success_rate = if total_results == 0 {
            0.0
        } else {
            success_results as f64 / total_results as f64
        };

        self.avg_tool_latency_ms = if latency_count == 0 {
            0
        } else {
            latency_sum / latency_count as u64
        };
    }

    /// 记录一次工具缓存命中。
    pub fn record_cache_hit(&mut self) {
        self.tool_cache_hits += 1;
    }

    /// 记录一次工具重试。
    pub fn record_retry(&mut self) {
        self.tool_retries += 1;
    }

    /// 记录一次循环检测触发。
    pub fn record_loop_detection(&mut self) {
        self.loop_detection_triggered += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chatvcode_llm::ToolResult;
    use serde_json::Value;

    fn usage(prompt: usize, completion: usize) -> TokenUsage {
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    #[test]
    fn token_usage_merge_sums_fields() {
        let mut a = usage(10, 20);
        let b = usage(5, 7);
        a.merge(&b);
        assert_eq!(a.prompt_tokens, 15);
        assert_eq!(a.completion_tokens, 27);
        assert_eq!(a.total_tokens, 42);
    }

    #[test]
    fn token_usage_merge_ignores_inconsistent_total() {
        let mut a = usage(10, 20);
        let mut b = usage(1, 2);
        b.total_tokens = 999; // 故意错误
        a.merge(&b);
        assert_eq!(a.total_tokens, 33);
    }

    #[test]
    fn token_usage_add_usage_accumulates() {
        let mut a = usage(0, 0);
        a.add_usage(3, 4);
        a.add_usage(2, 1);
        assert_eq!(a.prompt_tokens, 5);
        assert_eq!(a.completion_tokens, 5);
        assert_eq!(a.total_tokens, 10);
    }

    #[test]
    fn merge_step_accumulates_tokens_and_timing() {
        let mut metrics = AgentMetrics::default();
        let step = AgentStep {
            step_number: 1,
            state: AgentState::Thinking,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![],
            tool_results: vec![],
            duration_ms: 100,
            token_usage: usage(50, 10),
        };
        metrics.merge_step(&step);
        assert_eq!(metrics.total_input_tokens, 50);
        assert_eq!(metrics.total_output_tokens, 10);
        assert_eq!(metrics.thinking_time_ms, 100);
        assert_eq!(metrics.acting_time_ms, 0);
    }

    #[test]
    fn merge_step_counts_tool_calls_by_name() {
        let mut metrics = AgentMetrics::default();
        let step = AgentStep {
            step_number: 1,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![
                chatvcode_llm::ToolCall {
                    name: "read_file".into(),
                    arguments: Default::default(),
                    id: None,
                },
                chatvcode_llm::ToolCall {
                    name: "read_file".into(),
                    arguments: Default::default(),
                    id: None,
                },
                chatvcode_llm::ToolCall {
                    name: "grep_code".into(),
                    arguments: Default::default(),
                    id: None,
                },
            ],
            tool_results: vec![],
            duration_ms: 30,
            token_usage: usage(0, 0),
        };
        metrics.merge_step(&step);
        assert_eq!(metrics.tool_calls_by_name.get("read_file"), Some(&2));
        assert_eq!(metrics.tool_calls_by_name.get("grep_code"), Some(&1));
    }

    #[test]
    fn recompute_derived_empty_steps() {
        let mut metrics = AgentMetrics::default();
        metrics.recompute_derived(&[]);
        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.tool_success_rate, 0.0);
        assert_eq!(metrics.avg_tool_latency_ms, 0);
    }

    #[test]
    fn recompute_derived_success_rate_and_latency() {
        let mut metrics = AgentMetrics::default();
        let step = AgentStep {
            step_number: 1,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![],
            tool_results: vec![
                ToolResult::success(Value::Null),
                ToolResult::error("boom"),
            ],
            duration_ms: 40,
            token_usage: usage(0, 0),
        };
        metrics.recompute_derived(&[step]);
        assert_eq!(metrics.total_steps, 1);
        assert_eq!(metrics.tool_success_rate, 0.5);
        // 2 results sharing 40ms => avg 20ms
        assert_eq!(metrics.avg_tool_latency_ms, 20);
    }

    #[test]
    fn recompute_derived_ignores_thinking_step_latency() {
        let mut metrics = AgentMetrics::default();
        let thinking = AgentStep {
            step_number: 1,
            state: AgentState::Thinking,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![],
            tool_results: vec![ToolResult::success(Value::Null)],
            duration_ms: 999,
            token_usage: usage(0, 0),
        };
        metrics.recompute_derived(&[thinking]);
        assert_eq!(metrics.avg_tool_latency_ms, 0);
        // success rate still computed
        assert_eq!(metrics.tool_success_rate, 1.0);
    }

    #[test]
    fn record_counters_increment() {
        let mut metrics = AgentMetrics::default();
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_retry();
        metrics.record_loop_detection();
        assert_eq!(metrics.tool_cache_hits, 2);
        assert_eq!(metrics.tool_retries, 1);
        assert_eq!(metrics.loop_detection_triggered, 1);
    }

    #[test]
    fn merge_then_recompute_full_flow() {
        let mut metrics = AgentMetrics::default();
        let step1 = AgentStep {
            step_number: 1,
            state: AgentState::Thinking,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![],
            tool_results: vec![],
            duration_ms: 50,
            token_usage: usage(100, 20),
        };
        let step2 = AgentStep {
            step_number: 2,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![
                chatvcode_llm::ToolCall {
                    name: "read_file".into(),
                    arguments: Default::default(),
                    id: None,
                },
            ],
            tool_results: vec![ToolResult::success(Value::Null)],
            duration_ms: 30,
            token_usage: usage(200, 40),
        };
        let steps = vec![step1.clone(), step2.clone()];

        metrics.merge_step(&step1);
        metrics.merge_step(&step2);
        metrics.recompute_derived(&steps);

        assert_eq!(metrics.total_steps, 2);
        assert_eq!(metrics.total_input_tokens, 300);
        assert_eq!(metrics.total_output_tokens, 60);
        assert_eq!(metrics.thinking_time_ms, 50);
        assert_eq!(metrics.acting_time_ms, 30);
        assert_eq!(metrics.tool_calls_by_name.get("read_file"), Some(&1));
        assert_eq!(metrics.tool_success_rate, 1.0);
        assert_eq!(metrics.avg_tool_latency_ms, 30);
    }
}
