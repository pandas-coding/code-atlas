//! 用户确认机制：在关键操作前请求用户确认。
//!
//! 当 [`crate::types::AgentConfig::require_plan_confirmation`] 为 `true` 时，
//! Agent 在执行工具调用前会通过 [`ConfirmationHandler`] 询问用户是否继续。
//! 默认提供 [`AutoApproveHandler`]（自动批准）与 [`AlwaysDenyHandler`]，
//! 调用方（如 CLI）可实现自定义处理器以接入真实的交互式确认。

use crate::types::AgentStep;

/// 确认请求：描述需要用户确认的操作。
#[derive(Debug, Clone)]
pub struct ConfirmationRequest {
    /// 触发确认的步骤序号。
    pub step_number: usize,
    /// 即将执行的工具调用名称列表。
    pub tool_calls: Vec<String>,
    /// 当前已执行的步数。
    pub completed_steps: usize,
    /// 当前累计 token 使用量。
    pub total_tokens: usize,
}

/// 确认响应。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmationResponse {
    /// 批准执行。
    Approve,
    /// 批准并跳过后续所有确认（自动批准剩余步骤）。
    ApproveAll,
    /// 拒绝执行，Agent 应停止。
    Deny,
}

/// 确认处理器 trait：由调用方实现，在关键操作前请求用户确认。
///
/// 实现者可以通过返回 [`ConfirmationResponse::ApproveAll`] 来跳过后续
/// 所有确认请求，避免在长会话中反复打扰用户。
pub trait ConfirmationHandler: Send + Sync {
    /// 请求用户确认是否执行给定的操作。
    fn confirm(&self, request: &ConfirmationRequest) -> ConfirmationResponse;
}

/// 自动批准所有请求的处理器。
///
/// 行为等价于未设置确认处理器，但显式表达"自动批准"语义。
pub struct AutoApproveHandler;

impl ConfirmationHandler for AutoApproveHandler {
    fn confirm(&self, _request: &ConfirmationRequest) -> ConfirmationResponse {
        ConfirmationResponse::Approve
    }
}

/// 始终拒绝的处理器（用于测试或安全敏感场景）。
pub struct AlwaysDenyHandler;

impl ConfirmationHandler for AlwaysDenyHandler {
    fn confirm(&self, _request: &ConfirmationRequest) -> ConfirmationResponse {
        ConfirmationResponse::Deny
    }
}

/// 从 AgentStep 提取确认请求所需信息。
///
/// 返回 `None` 表示该步骤无需确认（如无工具调用）。
pub fn build_confirmation_request(step: &AgentStep, completed_steps: usize, total_tokens: usize) -> Option<ConfirmationRequest> {
    if step.tool_calls.is_empty() {
        return None;
    }
    Some(ConfirmationRequest {
        step_number: step.step_number,
        tool_calls: step.tool_calls.iter().map(|c| c.name.clone()).collect(),
        completed_steps,
        total_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentState, AgentStep, TokenUsage};
    use chatvcode_llm::ToolCall;
    use std::collections::HashMap;

    fn make_step_with_calls(step_number: usize, calls: Vec<ToolCall>) -> AgentStep {
        AgentStep {
            step_number,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: calls,
            tool_results: vec![],
            duration_ms: 0,
            token_usage: TokenUsage::default(),
        }
    }

    fn make_call(name: &str) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments: HashMap::new(),
            id: None,
        }
    }

    #[test]
    fn auto_approve_always_approves() {
        let handler = AutoApproveHandler;
        let req = ConfirmationRequest {
            step_number: 1,
            tool_calls: vec!["read_file".into()],
            completed_steps: 0,
            total_tokens: 100,
        };
        assert_eq!(handler.confirm(&req), ConfirmationResponse::Approve);
    }

    #[test]
    fn always_deny_always_denies() {
        let handler = AlwaysDenyHandler;
        let req = ConfirmationRequest {
            step_number: 1,
            tool_calls: vec!["read_file".into()],
            completed_steps: 0,
            total_tokens: 100,
        };
        assert_eq!(handler.confirm(&req), ConfirmationResponse::Deny);
    }

    #[test]
    fn build_request_from_step_with_calls() {
        let step = make_step_with_calls(2, vec![make_call("read_file"), make_call("grep_code")]);
        let req = build_confirmation_request(&step, 1, 500).unwrap();
        assert_eq!(req.step_number, 2);
        assert_eq!(req.tool_calls.len(), 2);
        assert_eq!(req.tool_calls[0], "read_file");
        assert_eq!(req.completed_steps, 1);
        assert_eq!(req.total_tokens, 500);
    }

    #[test]
    fn build_request_from_step_without_calls_returns_none() {
        let step = AgentStep {
            step_number: 1,
            state: AgentState::Thinking,
            thinking_phase: None,
            thought: None,
            tool_calls: vec![],
            tool_results: vec![],
            duration_ms: 0,
            token_usage: TokenUsage::default(),
        };
        assert!(build_confirmation_request(&step, 0, 0).is_none());
    }
}
