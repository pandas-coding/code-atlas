//! SSE（Server-Sent Events）事件序列化：把 [`AgentEvent`] 转换为可发送给
//! Web 客户端的文本帧。

use chatvcode_agent::types::AgentEvent;

/// 单个 SSE 事件的可测试表示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// 事件类型（对应 SSE 的 `event:` 行）。
    pub event: String,
    /// 事件数据（对应 SSE 的 `data:` 行，可为多行）。
    pub data: String,
    /// 可选的事件 ID（对应 SSE 的 `id:` 行）。
    pub id: Option<String>,
}

impl SseEvent {
    /// 创建一个普通事件。
    pub fn new(event: impl Into<String>, data: impl Into<String>) -> Self {
        Self { event: event.into(), data: data.into(), id: None }
    }

    /// 创建一个带 ID 的事件。
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// 错误事件。
    pub fn error(message: impl Into<String>) -> Self {
        Self::new("error", serde_json::json!({ "message": message.into() }).to_string())
    }

    /// 关闭事件（指示流结束）。
    pub fn close() -> Self {
        Self::new("done", "{\"reason\":\"stream-end\"}")
    }

    /// 序列化为 SSE 帧文本（以 `\n\n` 结尾）。
    pub fn to_frame(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("event: {}\n", self.event));
        if let Some(id) = &self.id {
            out.push_str(&format!("id: {}\n", id));
        }
        for line in self.data.split('\n') {
            out.push_str(&format!("data: {}\n", line));
        }
        out.push('\n');
        out
    }
}

/// 把 [`AgentEvent`] 适配为 [`SseEvent`]。
pub struct AgentEventSseAdapter;

impl AgentEventSseAdapter {
    /// 把 [`AgentEvent`] 转换为 SSE 帧文本。
    pub fn to_sse_frame(&self, event: &AgentEvent) -> String {
        self.to_sse_event(event).to_frame()
    }

    /// 把 [`AgentEvent`] 转换为 [`SseEvent`]。
    pub fn to_sse_event(&self, event: &AgentEvent) -> SseEvent {
        match event {
            AgentEvent::StateChanged { from, to } => SseEvent::new(
                "state_changed",
                serde_json::json!({ "from": format!("{:?}", from), "to": format!("{:?}", to) })
                    .to_string(),
            ),
            AgentEvent::ThinkingPhaseChanged { phase } => SseEvent::new(
                "thinking_phase_changed",
                serde_json::json!({ "phase": format!("{:?}", phase) }).to_string(),
            ),
            AgentEvent::Thinking { text } => {
                SseEvent::new("thinking", serde_json::json!({ "text": text }).to_string())
            }
            AgentEvent::ToolCallStarted { name, arguments } => SseEvent::new(
                "tool_call_started",
                serde_json::json!({ "name": name, "arguments": arguments }).to_string(),
            ),
            AgentEvent::ToolCallCompleted { name, result, cached } => SseEvent::new(
                "tool_call_completed",
                serde_json::json!({ "name": name, "result": result, "cached": cached })
                    .to_string(),
            ),
            AgentEvent::ToolCallFailed { name, error, will_retry } => SseEvent::new(
                "tool_call_failed",
                serde_json::json!({ "name": name, "error": error, "will_retry": will_retry })
                    .to_string(),
            ),
            AgentEvent::ToolCallRetrying { name, attempt, max_attempts } => SseEvent::new(
                "tool_call_retrying",
                serde_json::json!({ "name": name, "attempt": attempt, "max_attempts": max_attempts })
                    .to_string(),
            ),
            AgentEvent::StepCompleted { step } => SseEvent::new(
                "step_completed",
                serde_json::json!({ "step": step }).to_string(),
            ),
            AgentEvent::LoopDetected { detection_type } => SseEvent::new(
                "loop_detected",
                serde_json::json!({ "detection_type": detection_type }).to_string(),
            ),
            AgentEvent::AnswerStarted => SseEvent::new("answer_started", "{}"),
            AgentEvent::AnswerToken { text } => {
                SseEvent::new("answer_token", serde_json::json!({ "text": text }).to_string())
            }
            AgentEvent::AnswerCompleted { response } => SseEvent::new(
                "answer_completed",
                serde_json::to_string(response).unwrap_or_else(|_| "{}".into()),
            ),
            AgentEvent::Error { message } => SseEvent::error(message),
            AgentEvent::TokenBudgetWarning { used, remaining } => SseEvent::new(
                "token_budget_warning",
                serde_json::json!({ "used": used, "remaining": remaining }).to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chatvcode_agent::types::{AgentState, AgentResponse, AgentStopReason, ThinkingPhase};

    fn adapter() -> AgentEventSseAdapter {
        AgentEventSseAdapter
    }

    #[test]
    fn state_changed_uses_correct_event_name() {
        let ev = AgentEvent::StateChanged { from: AgentState::Thinking, to: AgentState::Acting };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "state_changed");
        assert!(sse.data.contains("Thinking"));
        assert!(sse.data.contains("Acting"));
    }

    #[test]
    fn thinking_phase_event() {
        let ev = AgentEvent::ThinkingPhaseChanged { phase: ThinkingPhase::Planning };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "thinking_phase_changed");
        assert!(sse.data.contains("Planning"));
    }

    #[test]
    fn thinking_event_is_typed() {
        let ev = AgentEvent::Thinking { text: "considering".into() };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "thinking");
        assert!(sse.data.contains("considering"));
    }

    #[test]
    fn tool_call_started_frame() {
        let ev = AgentEvent::ToolCallStarted {
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "tool_call_started");
        assert!(sse.data.contains("read_file"));
    }

    #[test]
    fn answer_token_uses_separate_events() {
        let ev1 = AgentEvent::AnswerToken { text: "He".into() };
        let ev2 = AgentEvent::AnswerToken { text: "llo".into() };
        let f1 = adapter().to_sse_frame(&ev1);
        let f2 = adapter().to_sse_frame(&ev2);
        assert!(f1.contains("answer_token"));
        assert!(f2.contains("answer_token"));
        assert!(f1.contains("He"));
        assert!(f2.contains("llo"));
    }

    #[test]
    fn answer_completed_serializes_full_response() {
        let resp = AgentResponse {
            answer: "done".into(),
            steps: vec![],
            total_token_usage: chatvcode_agent::types::TokenUsage::default(),
            total_duration_ms: 5,
            total_tool_calls: 0,
            stop_reason: AgentStopReason::Completed,
            metrics: chatvcode_agent::types::AgentMetrics::default(),
            sources: vec![],
            self_evaluation: None,
        };
        let ev = AgentEvent::AnswerCompleted { response: resp };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "answer_completed");
        assert!(sse.data.contains("done"));
    }

    #[test]
    fn error_event_uses_error_event_name() {
        let ev = AgentEvent::Error { message: "boom".into() };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "error");
        assert!(sse.data.contains("boom"));
    }

    #[test]
    fn token_budget_warning_event() {
        let ev = AgentEvent::TokenBudgetWarning { used: 100, remaining: 50 };
        let sse = adapter().to_sse_event(&ev);
        assert_eq!(sse.event, "token_budget_warning");
        assert!(sse.data.contains("100"));
        assert!(sse.data.contains("50"));
    }

    #[test]
    fn sse_frame_format_with_id() {
        let e = SseEvent::new("tick", "42").with_id("1");
        let f = e.to_frame();
        assert!(f.contains("id: 1\n"));
        assert!(f.contains("event: tick\n"));
        assert!(f.contains("data: 42\n"));
        assert!(f.ends_with("\n\n"));
    }

    #[test]
    fn sse_event_close_includes_done_event() {
        let e = SseEvent::close();
        assert_eq!(e.event, "done");
    }

    #[test]
    fn sse_event_multi_line_data_is_split() {
        let e = SseEvent::new("multi", "line1\nline2");
        let f = e.to_frame();
        assert!(f.contains("data: line1\n"));
        assert!(f.contains("data: line2\n"));
    }
}
