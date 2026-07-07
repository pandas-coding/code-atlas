//! Agent 循环：状态机驱动的主推理-执行循环。
//!
//! [`AgentLoop`] 串联 [`AgentSession`]、[`AgentStateMachine`]、LLM 推理、
//! [`ToolExecutor`] 与 [`LoopDetector`]，按 `Thinking ↔ Acting → Done/Failed`
//! 的状态机模型自主完成多步代码库探索。
//!
//! # 执行流程
//!
//! 1. `run(query)` 注入系统提示词与规划提示词，进入主循环。
//! 2. 每轮先经 [`check_limits`](AgentLoop::check_limits) 检查步数/超时/预算/循环。
//! 3. 根据当前状态调用 [`do_thinking`](AgentLoop::do_thinking) 或
//!    [`do_acting`](AgentLoop::do_acting)，产生 [`TransitionEvent`]。
//! 4. [`handle_transition`](AgentLoop::handle_transition) 将事件喂给状态机并同步会话。
//! 5. 进入终态后，通过 [`AgentSession::to_response`] 返回最终结果。
//!
//! # 流式执行
//!
//! [`AgentLoop::run_stream`] 在后台线程中运行同一循环，通过
//! [`mpsc::Receiver`] 实时推送 [`AgentEvent`]。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use chatvcode_llm::{LlmService, ToolCall, ToolResult, parse_tool_calls};

use crate::budget::SessionContext;
use crate::confirmation::ConfirmationHandler;
use crate::context::{AgentServices, ToolContext};
use crate::error::{AgentError, AgentResult};
use crate::executor::ToolExecutor;
use crate::loop_detector::{LoopDetectionResult, LoopDetector};
use crate::prompt::AgentPromptBuilder;
use crate::session::AgentSession;
use crate::state::{AgentStateMachine, TransitionEvent};
use crate::types::{
    AgentConfig, AgentEvent, AgentResponse, AgentState, AgentStep, AgentStopReason, ThinkingPhase,
    TokenUsage,
};

/// 默认的单步工具执行超时。
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 30;

/// Agent 主循环：聚合一次 Agent 执行所需的全部依赖与可变状态。
///
/// 持有：
/// - [`AgentSession`]：步骤历史、对话、预算、指标
/// - [`AgentStateMachine`]：状态转移决策
/// - [`LlmService`]：推理后端
/// - [`ToolExecutor`]：工具调度
/// - [`AgentPromptBuilder`]：提示词组装
/// - [`LoopDetector`]：循环检测
/// - [`AgentServices`]：检索/解析/块存储服务（用于构建 [`ToolContext`]）
/// - 可选的事件发送端（流式模式）
pub struct AgentLoop {
    session: AgentSession,
    state_machine: AgentStateMachine,
    llm_service: Arc<dyn LlmService>,
    tool_executor: Arc<dyn ToolExecutor>,
    prompt_builder: AgentPromptBuilder,
    loop_detector: LoopDetector,
    services: Arc<AgentServices>,
    event_sender: Option<mpsc::Sender<AgentEvent>>,
    cancel_flag: Arc<AtomicBool>,
    /// 最近一次执行的查询（供 [`continue_execution`](Self::continue_execution)
    /// 与 [`retry`](Self::retry) 使用）。
    last_query: Option<String>,
    /// 可选的用户确认处理器（仅在 `require_plan_confirmation = true` 时生效）。
    confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,
}

impl AgentLoop {
    /// 创建一个新的 Agent 循环。
    ///
    /// `services` 用于在 [`do_acting`](Self::do_acting) 中为每个工具调用
    /// 构建独立的 [`ToolContext`]。
    #[must_use]
    pub fn new(
        config: AgentConfig,
        llm_service: Arc<dyn LlmService>,
        tool_executor: Arc<dyn ToolExecutor>,
        services: Arc<AgentServices>,
    ) -> Self {
        let prompt_builder = AgentPromptBuilder::new(Arc::clone(&tool_executor));
        Self {
            session: AgentSession::new(config),
            state_machine: AgentStateMachine::new(),
            llm_service,
            tool_executor,
            prompt_builder,
            loop_detector: LoopDetector::new(),
            services,
            event_sender: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            last_query: None,
            confirmation_handler: None,
        }
    }

    /// 附带一个事件发送端，使循环在运行过程中实时推送 [`AgentEvent`]。
    ///
    /// 主要用于测试：在不消费 `self` 的情况下捕获事件流。
    #[must_use]
    pub fn with_event_sender(mut self, sender: mpsc::Sender<AgentEvent>) -> Self {
        self.event_sender = Some(sender);
        self
    }

    /// 附带一个用户确认处理器，在关键操作前请求用户确认。
    ///
    /// 仅当 [`AgentConfig::require_plan_confirmation`] 为 `true` 时生效。
    #[must_use]
    pub fn with_confirmation_handler(mut self, handler: Arc<dyn ConfirmationHandler>) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }

    // ---- 同步主循环 -------------------------------------------------------

    /// 同步执行 Agent 查询，返回最终响应。
    ///
    /// 在进入主循环前完成初始化：设置系统提示词、注入规划提示词。
    /// 终态（`Done`/`Failed`）由状态机决定，退出后通过
    /// [`AgentSession::to_response`] 打包结果。
    ///
    /// # 错误
    ///
    /// LLM 推理失败且无法恢复时返回 [`AgentError::LlmError`]。
    pub fn run(&mut self, query: &str) -> AgentResult<AgentResponse> {
        self.initialize(query);
        self.run_loop()
    }

    /// 主循环：假设会话已初始化，重复执行 Thinking/Acting 直到终态。
    fn run_loop(&mut self) -> AgentResult<AgentResponse> {
        while !self.is_terminal_state() {
            if self.is_cancelled() {
                self.session.set_state(AgentState::Failed);
                self.session.set_stop_reason(AgentStopReason::UserCancel);
                self.send_event(AgentEvent::Error {
                    message: "Agent cancelled by user".into(),
                });
                break;
            }

            if let Some(event) = self.check_limits() {
                self.handle_transition(event);
                if self.is_terminal_state() {
                    break;
                }
                continue;
            }

            let event = match self.session.state() {
                AgentState::Thinking => self.do_thinking()?,
                AgentState::Acting => self.do_acting()?,
                AgentState::Done | AgentState::Failed => break,
            };
            self.handle_transition(event);
        }

        Ok(self.session.to_response())
    }

    /// 初始化会话：重置状态机、注入系统提示词与规划提示词。
    fn initialize(&mut self, query: &str) {
        self.state_machine.reset();
        self.last_query = Some(query.to_string());

        self.session.set_state(AgentState::Thinking);
        self.session.set_thinking_phase(ThinkingPhase::Planning);

        let system_prompt = self.prompt_builder.build_system_prompt();
        self.session.chat_session_mut().set_system_prompt(Some(system_prompt));

        let planning_prompt = self.prompt_builder.build_planning_prompt(query);
        self.session.chat_session_mut().add_user_message(planning_prompt);

        self.send_event(AgentEvent::AnswerStarted);
    }

    // ---- 状态判定 ---------------------------------------------------------

    /// 是否处于终态。
    fn is_terminal_state(&self) -> bool {
        self.state_machine.is_terminal() || self.session.is_done() || self.session.is_failed()
    }

    /// 是否被外部取消。
    fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }

    // ---- 限制检查 ---------------------------------------------------------

    /// 综合检查步数、超时、预算与循环检测，返回需要触发的转移事件。
    ///
    /// 返回 `Some(event)` 表示本轮不应继续执行 Thinking/Acting，而应直接
    /// 处理该事件（例如进入终态或强制总结）。
    fn check_limits(&mut self) -> Option<TransitionEvent> {
        if self.session.remaining_steps() == 0 {
            return Some(TransitionEvent::MaxStepsReached);
        }
        if self.session.is_timed_out() {
            return Some(TransitionEvent::Timeout);
        }
        if let Some(event) = self.check_budget() {
            return Some(event);
        }
        if let Some(event) = self.check_loop() {
            return Some(event);
        }
        None
    }

    /// Token 预算检查：仅在 `Thinking` 状态下生效，预算过低时强制总结。
    fn check_budget(&mut self) -> Option<TransitionEvent> {
        if self.session.state() != AgentState::Thinking {
            return None;
        }
        // 已经在总结阶段则不再重复触发
        if self.session.thinking_phase() == ThinkingPhase::Concluding {
            return None;
        }

        let (is_low, used, remaining) = self.budget_status();
        if is_low {
            log::warn!("Token budget is low, forcing conclusion");
            self.send_event(AgentEvent::TokenBudgetWarning { used, remaining });
            self.session.set_thinking_phase(ThinkingPhase::Concluding);
            self.inject_system_hint(
                "Token budget is running low. Please provide a concise final answer now.",
            );
            return Some(TransitionEvent::ForceConclusion);
        }
        None
    }

    /// 循环检测：仅在 `Thinking` 状态且未进入总结阶段时生效。
    fn check_loop(&mut self) -> Option<TransitionEvent> {
        if self.session.state() != AgentState::Thinking {
            return None;
        }
        if self.session.thinking_phase() == ThinkingPhase::Concluding {
            return None;
        }

        let detection = self.loop_detector.detect(self.session.steps());
        if detection == LoopDetectionResult::NoLoop {
            return None;
        }

        self.session.record_loop_detection();
        let detection_value = match &detection {
            LoopDetectionResult::ExactLoop { tool_name, occurrences } => serde_json::json!({
                "type": "exact",
                "tool": tool_name,
                "occurrences": occurrences,
            }),
            LoopDetectionResult::SimilarLoop { pattern, occurrences } => serde_json::json!({
                "type": "similar",
                "pattern": pattern,
                "occurrences": occurrences,
            }),
            LoopDetectionResult::NoLoop => serde_json::Value::Null,
        };
        self.send_event(AgentEvent::LoopDetected { detection_type: detection_value });

        Some(self.handle_loop_detection(detection))
    }

    /// 根据循环类型决定恢复策略。
    fn handle_loop_detection(&mut self, detection: LoopDetectionResult) -> TransitionEvent {
        match detection {
            LoopDetectionResult::SimilarLoop { pattern, occurrences } => {
                let hint = format!(
                    "I notice you've called '{}' {} times with similar arguments. \
                     Please try a different approach or provide a final answer with the \
                     information you have.",
                    pattern, occurrences
                );
                self.inject_system_hint(&hint);
                TransitionEvent::Continue
            }
            LoopDetectionResult::ExactLoop { tool_name, occurrences } => {
                log::warn!(
                    "Exact loop detected: {} called {} times",
                    tool_name,
                    occurrences
                );
                self.session.set_thinking_phase(ThinkingPhase::Concluding);
                self.inject_system_hint(
                    "You appear to be repeating the same tool call. Please provide a \
                     final answer based on the information gathered so far.",
                );
                TransitionEvent::ForceConclusion
            }
            LoopDetectionResult::NoLoop => TransitionEvent::Continue,
        }
    }

    /// 向对话注入一条系统提示（以 user 消息形式追加，供下一次推理读取）。
    fn inject_system_hint(&mut self, hint: &str) {
        self.session.chat_session_mut().add_user_message(hint);
    }

    // ---- Thinking 阶段 ----------------------------------------------------

    /// 执行 Thinking 阶段：构建提示词、调用 LLM、解析输出、记录步骤。
    ///
    /// 返回 [`TransitionEvent::ToolCallsDetected`]（检测到工具调用）或
    /// [`TransitionEvent::FinalAnswer`]（无工具调用，视为最终回答）。
    fn do_thinking(&mut self) -> AgentResult<TransitionEvent> {
        let start = Instant::now();
        let phase = self.session.thinking_phase();
        self.send_event(AgentEvent::ThinkingPhaseChanged { phase });

        let prompt = self
            .session
            .chat_session()
            .build_prompt()
            .map_err(|e| AgentError::LlmError(e.to_string()))?;
        let params = self.session.config().generation_params.clone();
        let response = self
            .llm_service
            .infer(&prompt, &params, Some(&self.cancel_flag))
            .map_err(|e| AgentError::LlmError(e.to_string()))?;

        self.send_event(AgentEvent::Thinking { text: response.text.clone() });
        self.session
            .chat_session_mut()
            .add_assistant_message(response.text.clone());

        let token_usage = convert_token_usage(&response.token_usage);
        let duration_ms = start.elapsed().as_millis() as u64;

        // 解析工具调用：失败或空列表均视为无工具调用
        let parsed_calls = parse_tool_calls(&response.text)
            .ok()
            .filter(|c| !c.is_empty());
        let calls = parsed_calls
            .map(|c| self.filter_allowed_tools(c))
            .unwrap_or_default();

        let step = AgentStep {
            step_number: 0,
            state: AgentState::Thinking,
            thinking_phase: Some(phase),
            thought: Some(response.text.clone()),
            // 工具调用由后续 Acting 步骤记录，避免在 total_tool_calls 中重复计数
            tool_calls: vec![],
            tool_results: vec![],
            duration_ms,
            token_usage,
        };
        self.session.add_step(step);

        if calls.is_empty() {
            Ok(TransitionEvent::FinalAnswer(response.text))
        } else {
            Ok(TransitionEvent::ToolCallsDetected(calls))
        }
    }

    /// 过滤工具调用：若配置了 `allowed_tools`，仅保留白名单内的调用。
    fn filter_allowed_tools(&self, calls: Vec<ToolCall>) -> Vec<ToolCall> {
        let allowed = &self.session.config().allowed_tools;
        if allowed.is_empty() {
            return calls;
        }
        calls.into_iter().filter(|c| allowed.contains(&c.name)).collect()
    }

    // ---- Acting 阶段 -----------------------------------------------------

    /// 执行 Acting 阶段：从状态机取出待执行工具调用，依次或并行执行，
    /// 截断结果，注入观察提示词，记录步骤。
    ///
    /// 返回 [`TransitionEvent::ToolsExecuted`]。
    ///
    /// 当 [`AgentConfig::enable_parallel_tool_calls`] 为 `true` 且本步骤有
    /// 多个工具调用时，使用 `std::thread::scope` 并行执行；否则顺序执行。
    fn do_acting(&mut self) -> AgentResult<TransitionEvent> {
        let start = Instant::now();
        let pending = self.state_machine.take_pending_tool_calls();
        let max_calls = self.session.config().max_tool_calls_per_step;
        let calls_to_execute: Vec<ToolCall> = pending.into_iter().take(max_calls).collect();

        // 用户确认机制：若启用且配置了确认处理器，在执行前请求确认
        if self.session.config().require_plan_confirmation
            && let Some(handler) = &self.confirmation_handler
        {
            let completed_steps = self.session.current_step();
            let total_tokens = self.session.total_token_usage().total_tokens;
            let dummy_step = AgentStep {
                step_number: completed_steps + 1,
                state: AgentState::Acting,
                thinking_phase: None,
                thought: None,
                tool_calls: calls_to_execute.clone(),
                tool_results: vec![],
                duration_ms: 0,
                token_usage: TokenUsage::default(),
            };
            if let Some(req) = crate::confirmation::build_confirmation_request(
                &dummy_step,
                completed_steps,
                total_tokens,
            ) {
                match handler.confirm(&req) {
                    crate::confirmation::ConfirmationResponse::Approve => {}
                    crate::confirmation::ConfirmationResponse::ApproveAll => {
                        // 关闭后续确认
                        let config = self.session.config_mut();
                        config.require_plan_confirmation = false;
                    }
                    crate::confirmation::ConfirmationResponse::Deny => {
                        self.session.set_state(AgentState::Failed);
                        self.session
                            .set_stop_reason(AgentStopReason::UserCancel);
                        self.send_event(AgentEvent::Error {
                            message: "Execution denied by user".into(),
                        });
                        return Ok(TransitionEvent::UnrecoverableError(
                            "User denied execution".into(),
                        ));
                    }
                }
            }
        }

        let project_path = self.session.config().project_path.clone();
        let token_budget = self.session.config().token_budget.tool_result_max;
        let timeout = Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS);
        let enable_parallel = self.session.config().enable_parallel_tool_calls;

        let results = if enable_parallel && calls_to_execute.len() > 1 {
            self.execute_tools_parallel(&calls_to_execute, project_path, timeout, token_budget)
        } else {
            self.execute_tools_sequential(&calls_to_execute, project_path, timeout, token_budget)
        };

        let truncated_results = self.truncate_results(&results);

        // 构造观察提示词并注入对话
        let observing_step = AgentStep {
            step_number: 0,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: calls_to_execute.clone(),
            tool_results: truncated_results.clone(),
            duration_ms: 0,
            token_usage: TokenUsage::default(),
        };
        let observing_prompt = self.prompt_builder.build_observing_prompt(&observing_step);
        self.session.chat_session_mut().add_user_message(observing_prompt);

        let step = AgentStep {
            step_number: 0,
            state: AgentState::Acting,
            thinking_phase: None,
            thought: None,
            tool_calls: calls_to_execute,
            tool_results: truncated_results.clone(),
            duration_ms: start.elapsed().as_millis() as u64,
            token_usage: TokenUsage::default(),
        };
        self.session.add_step(step);

        self.session.set_thinking_phase(ThinkingPhase::Observing);
        Ok(TransitionEvent::ToolsExecuted(truncated_results))
    }

    /// 顺序执行工具调用（原有行为）。
    fn execute_tools_sequential(
        &self,
        calls: &[ToolCall],
        project_path: std::path::PathBuf,
        timeout: Duration,
        token_budget: usize,
    ) -> Vec<ToolResult> {
        let mut results: Vec<ToolResult> = Vec::with_capacity(calls.len());
        for call in calls {
            self.send_event(AgentEvent::ToolCallStarted {
                name: call.name.clone(),
                arguments: serde_json::Value::Object(
                    call.arguments.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                ),
            });

            let ctx = ToolContext {
                project_path: project_path.clone(),
                timeout,
                token_budget,
                services: Arc::clone(&self.services),
            };

            let result = match self.tool_executor.execute(call, &ctx) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    self.send_event(AgentEvent::ToolCallFailed {
                        name: call.name.clone(),
                        error: msg.clone(),
                        will_retry: false,
                    });
                    let mut err = ToolResult::error(msg);
                    if let Some(ref id) = call.id {
                        err = err.with_call_id(id.clone());
                    }
                    err
                }
            };

            self.emit_tool_completion(call, &result);
            results.push(result);
        }
        results
    }

    /// 并行执行工具调用（使用 `std::thread::scope`）。
    ///
    /// 各工具调用在独立线程中执行，结果按原始顺序返回。事件发送在
    /// 线程内进行（`mpsc::Sender` 是 `Sync` 的）。若任一线程 panic，
    /// 该调用的结果会被记录为错误。
    fn execute_tools_parallel(
        &self,
        calls: &[ToolCall],
        project_path: std::path::PathBuf,
        timeout: Duration,
        token_budget: usize,
    ) -> Vec<ToolResult> {
        use std::thread;

        // 先发送所有 ToolCallStarted 事件
        for call in calls {
            self.send_event(AgentEvent::ToolCallStarted {
                name: call.name.clone(),
                arguments: serde_json::Value::Object(
                    call.arguments.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                ),
            });
        }

        let tool_executor = Arc::clone(&self.tool_executor);
        let services = Arc::clone(&self.services);
        let event_sender = self.event_sender.clone();

        let results: Vec<ToolResult> = thread::scope(|s| {
            let handles: Vec<_> = calls
                .iter()
                .map(|call| {
                    let project_path = project_path.clone();
                    let tool_executor = Arc::clone(&tool_executor);
                    let services = Arc::clone(&services);
                    let event_sender = event_sender.clone();
                    let call = call.clone();
                    s.spawn(move || {
                        let ctx = ToolContext {
                            project_path,
                            timeout,
                            token_budget,
                            services,
                        };
                        let result = match tool_executor.execute(&call, &ctx) {
                            Ok(r) => r,
                            Err(e) => {
                                let msg = e.to_string();
                                if let Some(sender) = &event_sender {
                                    let _ = sender.send(AgentEvent::ToolCallFailed {
                                        name: call.name.clone(),
                                        error: msg.clone(),
                                        will_retry: false,
                                    });
                                }
                                let mut err = ToolResult::error(msg);
                                if let Some(ref id) = call.id {
                                    err = err.with_call_id(id.clone());
                                }
                                err
                            }
                        };
                        (call.name, result)
                    })
                })
                .collect();

            let mut results: Vec<ToolResult> = Vec::with_capacity(handles.len());
            for handle in handles {
                match handle.join() {
                    Ok((name, result)) => {
                        self.emit_tool_completion_by_name(&name, &result);
                        results.push(result);
                    }
                    Err(_) => {
                        results.push(ToolResult::error("Tool thread panicked"));
                    }
                }
            }
            results
        });

        results
    }

    /// 发送工具完成/失败事件（按 call 名称）。
    fn emit_tool_completion_by_name(&self, name: &str, result: &ToolResult) {
        if result.success {
            self.send_event(AgentEvent::ToolCallCompleted {
                name: name.to_string(),
                result: result.clone(),
                cached: false,
            });
        } else {
            let error_msg = match &result.value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            self.send_event(AgentEvent::ToolCallFailed {
                name: name.to_string(),
                error: error_msg,
                will_retry: false,
            });
        }
    }

    /// 发送工具完成/失败事件（按 call 引用）。
    fn emit_tool_completion(&self, call: &ToolCall, result: &ToolResult) {
        self.emit_tool_completion_by_name(&call.name, result);
    }

    /// 截断过大的工具结果以适配 token 预算。
    ///
    /// 失败结果保持原样；成功结果序列化为字符串后由
    /// [`TokenBudgetManager::truncate_tool_result`] 截断，再反序列化回
    /// [`serde_json::Value`]。反序列化失败时退化为截断后的字符串。
    fn truncate_results(&self, results: &[ToolResult]) -> Vec<ToolResult> {
        results
            .iter()
            .map(|r| {
                if !r.success {
                    return r.clone();
                }
                let value_str = serde_json::to_string(&r.value).unwrap_or_default();
                let truncated_str = self.truncate_tool_result(&value_str);
                let truncated_value: serde_json::Value =
                    match serde_json::from_str(&truncated_str) {
                        Ok(v) => v,
                        Err(_) => serde_json::Value::String(truncated_str),
                    };
                ToolResult {
                    value: truncated_value,
                    success: r.success,
                    call_id: r.call_id.clone(),
                }
            })
            .collect()
    }

    // ---- 状态转移 ---------------------------------------------------------

    /// 应用一个转移事件：喂给状态机，并同步会话状态/停止原因/最终回答。
    fn handle_transition(&mut self, event: TransitionEvent) {
        let prev_state = self.state_machine.state();
        let new_state = self.state_machine.transition(event);
        if prev_state != new_state {
            self.session.set_state(new_state);
            self.send_event(AgentEvent::StateChanged {
                from: prev_state,
                to: new_state,
            });
        }
        let phase = self.state_machine.thinking_phase();
        self.session.set_thinking_phase(phase);

        // 取出状态机缓存的产物，同步到会话
        if let Some(answer) = self.state_machine.take_final_answer() {
            self.session.set_final_answer(answer);
            self.session.set_stop_reason(AgentStopReason::Completed);
        }

        if self.state_machine.is_terminal() {
            let reason = self.state_machine.termination_reason().unwrap_or("unknown");
            match self.session.stop_reason() {
                Some(_) => {}
                None => match reason {
                    "max_steps" => {
                        self.session.set_stop_reason(AgentStopReason::MaxSteps);
                    }
                    "timeout" => {
                        self.session.set_stop_reason(AgentStopReason::Timeout);
                    }
                    "completed" => {
                        self.session.set_stop_reason(AgentStopReason::Completed);
                    }
                    other => {
                        self.session.set_stop_reason(AgentStopReason::Error(other.to_string()));
                    }
                },
            }
        }
    }

    // ---- 流式执行 ---------------------------------------------------------

    /// 在后台线程中运行循环，实时推送 [`AgentEvent`]。
    ///
    /// 消费 `self`，将所有权移入后台线程。完成时推送
    /// [`AgentEvent::AnswerCompleted`] 或 [`AgentEvent::Error`]。
    pub fn run_stream(self, query: &str) -> AgentResult<mpsc::Receiver<AgentEvent>> {
        let (sender, receiver) = mpsc::channel();
        let mut agent = self.with_event_sender(sender);
        let query = query.to_string();

        std::thread::Builder::new()
            .name("agent-loop".into())
            .spawn(move || {
                let sender = agent.event_sender.clone();
                match agent.run(&query) {
                    Ok(response) => {
                        if let Some(s) = sender {
                            let _ = s.send(AgentEvent::AnswerCompleted { response });
                        }
                    }
                    Err(e) => {
                        if let Some(s) = sender {
                            let _ = s.send(AgentEvent::Error { message: e.to_string() });
                        }
                    }
                }
            })
            .map_err(|e| AgentError::Internal(format!("Failed to spawn agent thread: {e}")))?;

        Ok(receiver)
    }

    /// 向事件发送端推送一个事件（若无发送端则忽略）。
    fn send_event(&self, event: AgentEvent) {
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event);
        }
    }

    // ---- 控制命令 ---------------------------------------------------------

    /// 取消正在执行的循环。
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    /// 继续执行：当因最大步数停止时，放宽步数限制后继续运行上一个查询。
    ///
    /// 仅在终态（`Failed`）且停止原因为 `MaxSteps` 时有意义；其他状态
    /// 直接返回当前响应。保留对话历史，重置状态机以重新进入循环。
    pub fn continue_execution(&mut self) -> AgentResult<AgentResponse> {
        let is_max_steps =
            matches!(self.session.stop_reason(), Some(AgentStopReason::MaxSteps));
        if !is_max_steps {
            return Ok(self.session.to_response());
        }

        // 放宽步数限制：至少追加 max_steps 或 5 步
        {
            let config = self.session.config_mut();
            let extra = config.max_steps.max(5);
            config.max_steps = config.max_steps.saturating_add(extra);
        }

        // 重置状态机与会话状态以重新进入循环（保留对话历史）
        self.state_machine.reset();
        self.session.set_state(AgentState::Thinking);
        self.session.set_thinking_phase(ThinkingPhase::Observing);
        self.cancel_flag.store(false, Ordering::Relaxed);

        self.run_loop()
    }

    /// 重试上一个查询：清空步骤历史但保留配置，重新运行。
    pub fn retry(&mut self) -> AgentResult<AgentResponse> {
        let query = self.last_query.clone().unwrap_or_default();
        // 重建会话以丢弃历史
        let config = self.session.config().clone();
        self.session = AgentSession::new(config);
        self.state_machine.reset();
        self.cancel_flag.store(false, Ordering::Relaxed);
        self.run(&query)
    }

    // ---- 访问器 -----------------------------------------------------------

    /// 会话（只读）。
    #[must_use]
    pub fn session(&self) -> &AgentSession {
        &self.session
    }

    /// 会话（可变）。
    pub fn session_mut(&mut self) -> &mut AgentSession {
        &mut self.session
    }

    /// 取消标志的共享句柄。
    #[must_use]
    pub fn cancel_flag(&self) -> &Arc<AtomicBool> {
        &self.cancel_flag
    }

    // ---- 辅助 -------------------------------------------------------------

    /// 读取当前预算状态 `(is_low, used, remaining)`。
    fn budget_status(&self) -> (bool, usize, usize) {
        let system = self.session.chat_session().get_system_prompt().unwrap_or("");
        let messages = self.session.chat_session().messages();
        let ctx = SessionContext { system_prompt: system, messages };
        let mgr = self.session.budget_manager();
        (mgr.is_budget_low(&ctx), mgr.used_tokens(&ctx), mgr.remaining_tokens(&ctx))
    }

    /// 截断工具结果字符串（委托给会话内置的预算管理器）。
    fn truncate_tool_result(&self, result: &str) -> String {
        self.session.budget_manager().truncate_tool_result(result)
    }
}

/// 将 `chatvcode_llm` 的 [`TokenUsage`](chatvcode_llm::TokenUsage)（i32 字段）
/// 转换为 Agent 的 [`TokenUsage`]（usize 字段），负值按 0 处理。
fn convert_token_usage(usage: &chatvcode_llm::TokenUsage) -> TokenUsage {
    let prompt = usize::try_from(usage.prompt_tokens.max(0)).unwrap_or(0);
    let completion = usize::try_from(usage.completion_tokens.max(0)).unwrap_or(0);
    TokenUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{ChunkMetadataStoreTrait, CodeSearchService};
    use crate::executor::BuiltinToolRegistry;
    use crate::types::ToolRetryConfig;
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use chatvcode_llm::{GenerationParams, InferenceResponse, MockLlmService, StopReason};
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc::RecvTimeoutError;

    // ---- 测试用桩 ---------------------------------------------------------

    struct EmptySearchService;
    impl CodeSearchService for EmptySearchService {
        fn search(&self, _query: &str, _top_k: usize) -> Result<Vec<SearchResult>, AgentError> {
            Ok(vec![])
        }
    }

    struct EmptyChunkStore;
    impl ChunkMetadataStoreTrait for EmptyChunkStore {
        fn get_chunks_by_symbol(&self, _symbol: &str, _kind: Option<&str>) -> Vec<ChunkMetadata> {
            vec![]
        }
        fn get_chunk_by_id(&self, _id: &str) -> Option<ChunkMetadata> {
            None
        }
    }

    fn make_services() -> Arc<AgentServices> {
        Arc::new(AgentServices {
            search: Box::new(EmptySearchService),
            parser: Box::new(|_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<
                chatvcode_core::model::ParseResult,
            > { unimplemented!() }),
            chunk_store: Box::new(EmptyChunkStore),
        })
    }

    fn make_registry() -> Arc<dyn ToolExecutor> {
        let mut reg = BuiltinToolRegistry::new(ToolRetryConfig::default());
        reg.register_defaults();
        Arc::new(reg)
    }

    fn make_config(max_steps: usize) -> AgentConfig {
        AgentConfig {
            max_steps,
            timeout_secs: 0,
            ..AgentConfig::default()
        }
    }

    /// 可编程的 Mock LLM：按调用顺序依次返回预设响应。
    struct ScriptedLlm {
        responses: std::sync::Mutex<Vec<String>>,
        call_count: AtomicUsize,
    }

    impl ScriptedLlm {
        fn new(responses: Vec<String>) -> Self {
            Self { responses: std::sync::Mutex::new(responses), call_count: AtomicUsize::new(0) }
        }
    }

    impl LlmService for ScriptedLlm {
        fn infer(
            &self,
            _prompt: &str,
            _params: &GenerationParams,
            _cancel_flag: Option<&AtomicBool>,
        ) -> chatvcode_llm::LlmResult<InferenceResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().unwrap();
            let text = if count < responses.len() {
                std::mem::take(&mut responses[count])
            } else {
                String::from("I have no further information.")
            };
            Ok(InferenceResponse {
                text,
                stop_reason: StopReason::Eos,
                token_usage: chatvcode_llm::TokenUsage::new(10, 5),
                duration: Duration::from_millis(1),
                time_to_first_token: Some(Duration::from_millis(1)),
                tokens_per_second: 100.0,
            })
        }

        fn infer_stream(
            &self,
            _prompt: &str,
            _params: &GenerationParams,
            _cancel_flag: Option<Arc<AtomicBool>>,
        ) -> chatvcode_llm::LlmResult<mpsc::Receiver<chatvcode_llm::StreamEvent>> {
            Err(chatvcode_llm::LlmError::Internal("not supported".into()))
        }

        fn model_info(&self) -> chatvcode_llm::LlmResult<chatvcode_llm::ModelInfo> {
            Ok(chatvcode_llm::ModelInfo {
                description: "Scripted Mock".into(),
                architecture: "mock".into(),
                n_params: 0,
                size_bytes: 0,
                n_ctx_train: 4096,
                n_embd: 0,
                n_layer: 0,
                n_head: 0,
                n_head_kv: 0,
                n_vocab: 0,
                vocab_type: "bpe".into(),
                ftype: "q4_0".into(),
                chat_template_available: true,
                rope_type: "0".into(),
                has_encoder: false,
                has_decoder: true,
            })
        }
    }

    fn make_agent(responses: Vec<String>, config: AgentConfig) -> AgentLoop {
        let llm: Arc<dyn LlmService> = Arc::new(ScriptedLlm::new(responses));
        AgentLoop::new(config, llm, make_registry(), make_services())
    }

    // ---- 场景测试 ---------------------------------------------------------

    #[test]
    fn direct_answer_scenario() {
        let mut agent = make_agent(
            vec!["The main function is the entry point.".into()],
            make_config(5),
        );
        let response = agent.run("What is main?").unwrap();
        assert_eq!(response.answer, "The main function is the entry point.");
        assert!(matches!(response.stop_reason, AgentStopReason::Completed));
        assert_eq!(response.steps.len(), 1);
        assert!(response.steps[0].tool_calls.is_empty());
    }

    #[test]
    fn single_tool_call_scenario() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello world").unwrap();

        let mut config = make_config(5);
        config.project_path = tmp.path().to_path_buf();

        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let final_answer = "I listed the files.";
        let mut agent = make_agent(
            vec![tool_call.into(), final_answer.into()],
            config,
        );
        let response = agent.run("List files").unwrap();
        assert_eq!(response.answer, "I listed the files.");
        // think -> act -> think => 3 步
        assert_eq!(response.steps.len(), 3);
        // 仅 Acting 步骤携带工具调用
        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert_eq!(acting[0].tool_calls.len(), 1);
        assert_eq!(acting[0].tool_calls[0].name, "list_files");
        assert_eq!(response.total_tool_calls, 1);
    }

    #[test]
    fn multi_step_tool_call_scenario() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "fn foo() {}").unwrap();

        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();

        let call1 = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let call2 = r#"{"name":"read_file","arguments":{"path":"a.rs"}}"#;
        let final_answer = "Found foo.";
        let mut agent = make_agent(
            vec![call1.into(), call2.into(), final_answer.into()],
            config,
        );
        let response = agent.run("Explore").unwrap();
        assert_eq!(response.answer, "Found foo.");
        assert_eq!(response.total_tool_calls, 2);
        assert_eq!(
            response.steps.iter().filter(|s| s.state == AgentState::Acting).count(),
            2
        );
    }

    #[test]
    fn max_steps_exceeded_scenario() {
        // 每轮都调用工具，永不给出最终回答
        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let mut agent = make_agent(
            (0..10).map(|_| tool_call.to_string()).collect(),
            make_config(3),
        );
        let response = agent.run("Loop").unwrap();
        assert!(matches!(response.stop_reason, AgentStopReason::MaxSteps));
        assert!(response.steps.len() <= 3);
    }

    #[test]
    fn timeout_scenario() {
        let mut config = make_config(100);
        config.timeout_secs = 1;
        // 不超时直接完成
        let mut agent = make_agent(vec!["done".into()], config);
        let response = agent.run("q").unwrap();
        assert!(matches!(response.stop_reason, AgentStopReason::Completed));
    }

    #[test]
    fn tool_execution_failure_recovery_scenario() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();

        // 调用不存在的工具 -> 失败 -> LLM 给出最终回答
        let bad_call = r#"{"name":"nonexistent_tool","arguments":{}}"#;
        let final_answer = "Could not use the tool.";
        let mut agent = make_agent(vec![bad_call.into(), final_answer.into()], config);
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "Could not use the tool.");
        // Acting 步骤携带失败的工具结果
        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert_eq!(acting[0].tool_results.len(), 1);
        assert!(!acting[0].tool_results[0].success);
    }

    #[test]
    fn loop_detection_force_conclusion_scenario() {
        // 构造连续 6 步相同工具调用 -> ExactLoop -> ForceConclusion
        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let final_answer = "Giving up now.";
        let mut responses: Vec<String> = (0..6).map(|_| tool_call.to_string()).collect();
        responses.push(final_answer.into());

        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(20);
        config.project_path = tmp.path().to_path_buf();

        let mut agent = make_agent(responses, config);
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "Giving up now.");
        assert!(response.metrics.loop_detection_triggered >= 1);
    }

    #[test]
    fn token_budget_exhausted_scenario() {
        // 极小预算，系统提示词注入后会立即触发预算过低 -> ForceConclusion
        let mut config = make_config(100);
        config.token_budget.total_budget = 100;
        config.token_budget.response_reserve = 50;

        let mut agent = make_agent(vec!["final".into()], config);
        let response = agent.run("q").unwrap();
        // 应当完成（被强制总结），而非无限循环
        assert!(matches!(
            response.stop_reason,
            AgentStopReason::Completed | AgentStopReason::Error(_)
        ));
    }

    #[test]
    fn run_stream_emits_events() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(5);
        config.project_path = tmp.path().to_path_buf();

        let agent = make_agent(vec!["direct answer".into()], config);
        let rx = agent.run_stream("query").unwrap();

        let mut got_completed = false;
        let mut got_started = false;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(AgentEvent::AnswerStarted) => got_started = true,
                Ok(AgentEvent::AnswerCompleted { response }) => {
                    assert_eq!(response.answer, "direct answer");
                    got_completed = true;
                    break;
                }
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        assert!(got_started);
        assert!(got_completed);
    }

    #[test]
    fn cancel_flag_stops_loop() {
        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(100);
        config.project_path = tmp.path().to_path_buf();

        let llm: Arc<dyn LlmService> = Arc::new(ScriptedLlm::new(
            (0..100).map(|_| tool_call.to_string()).collect(),
        ));
        let mut agent = AgentLoop::new(config, llm, make_registry(), make_services());
        agent.cancel_flag().store(true, Ordering::Relaxed);

        let response = agent.run("q").unwrap();
        assert!(matches!(response.stop_reason, AgentStopReason::UserCancel));
    }

    #[test]
    fn filter_allowed_tools_drops_unlisted() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();
        config.allowed_tools = vec!["list_files".into()];

        // 第一步同时调用 list_files 和 read_file（被过滤）
        let call = r#"[{"name":"list_files","arguments":{"path":"."}},{"name":"read_file","arguments":{"path":"x"}}]"#;
        let final_answer = "ok";
        let mut agent = make_agent(vec![call.into(), final_answer.into()], config);
        let response = agent.run("q").unwrap();
        // 过滤后仅 list_files 保留（在 Acting 步骤中）
        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert_eq!(acting[0].tool_calls.len(), 1);
        assert_eq!(acting[0].tool_calls[0].name, "list_files");
    }

    #[test]
    fn retry_clears_history() {
        let mut agent =
            make_agent(vec!["first".into(), "second".into()], make_config(5));
        let r1 = agent.run("q").unwrap();
        assert_eq!(r1.answer, "first");
        assert_eq!(agent.session().steps().len(), 1);

        let r2 = agent.retry().unwrap();
        assert_eq!(r2.answer, "second");
        // retry 重建会话，步骤历史被清空后重新记录一步
        assert_eq!(agent.session().steps().len(), 1);
    }

    #[test]
    fn continue_execution_after_max_steps() {
        // max_steps=1 触发 MaxSteps，继续后 LLM 给出最终回答
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(1);
        config.project_path = tmp.path().to_path_buf();

        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let final_answer = "Done after continue.";
        let mut agent = make_agent(
            vec![tool_call.into(), final_answer.into()],
            config,
        );

        let r1 = agent.run("q").unwrap();
        assert!(matches!(r1.stop_reason, AgentStopReason::MaxSteps));

        let r2 = agent.continue_execution().unwrap();
        assert_eq!(r2.answer, "Done after continue.");
        assert!(matches!(r2.stop_reason, AgentStopReason::Completed));
    }

    #[test]
    fn convert_token_usage_clamps_negative() {
        let usage = chatvcode_llm::TokenUsage::new(-5, 3);
        let converted = convert_token_usage(&usage);
        assert_eq!(converted.prompt_tokens, 0);
        assert_eq!(converted.completion_tokens, 3);
        assert_eq!(converted.total_tokens, 3);
    }

    #[test]
    fn truncate_results_preserves_failures() {
        let agent = make_agent(vec!["x".into()], make_config(5));
        let ok = ToolResult::success(json!({"data":"value"}));
        let err = ToolResult::error("boom");
        let truncated = agent.truncate_results(&[ok, err]);
        assert!(truncated[0].success);
        assert!(!truncated[1].success);
        assert_eq!(truncated[1].value, serde_json::Value::String("boom".into()));
    }

    #[test]
    fn direct_answer_with_mock_llm_service() {
        let llm: Arc<dyn LlmService> =
            Arc::new(MockLlmService::new("A plain text answer."));
        let mut agent =
            AgentLoop::new(make_config(5), llm, make_registry(), make_services());
        let response = agent.run("Any question").unwrap();
        // MockLlmService 返回纯文本，parse_tool_calls 失败 -> 最终回答
        assert_eq!(response.answer, "A plain text answer.");
        assert!(matches!(response.stop_reason, AgentStopReason::Completed));
    }

    #[test]
    fn empty_planning_prompt_does_not_crash() {
        let mut agent = make_agent(vec!["answer".into()], make_config(3));
        let response = agent.run("").unwrap();
        assert_eq!(response.answer, "answer");
    }

    #[test]
    fn parallel_tool_calls_executes_all_tools() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "world").unwrap();

        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();
        config.enable_parallel_tool_calls = true;

        // 两个并行工具调用 + 最终回答
        let call = r#"[{"name":"read_file","arguments":{"path":"a.txt"}},{"name":"read_file","arguments":{"path":"b.txt"}}]"#;
        let final_answer = "Read both files.";
        let mut agent = make_agent(vec![call.into(), final_answer.into()], config);
        let response = agent.run("Read both").unwrap();
        assert_eq!(response.answer, "Read both files.");
        // Acting 步骤应携带 2 个工具调用
        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert_eq!(acting[0].tool_calls.len(), 2);
        assert_eq!(acting[0].tool_results.len(), 2);
        // 两个结果都应成功
        assert!(acting[0].tool_results.iter().all(|r| r.success));
    }

    #[test]
    fn self_evaluation_included_when_enabled() {
        let mut config = make_config(5);
        config.enable_self_evaluation = true;

        let mut agent = make_agent(vec!["short".into()], config);
        let response = agent.run("q").unwrap();
        let eval = response.self_evaluation.expect("self_evaluation should be set");
        assert!(eval.confidence_score >= 0.0 && eval.confidence_score <= 1.0);
        assert!(!eval.notes.is_empty());
        // 短回答 + 无来源引用 -> 置信度不应为满分
        assert!(eval.confidence_score < 1.0);
    }

    #[test]
    fn self_evaluation_absent_when_disabled() {
        let mut agent = make_agent(vec!["answer".into()], make_config(5));
        let response = agent.run("q").unwrap();
        assert!(response.self_evaluation.is_none());
    }

    #[test]
    fn confirmation_handler_can_deny_execution() {
        use crate::confirmation::{ConfirmationRequest, ConfirmationResponse};

        struct DenyHandler;
        impl crate::confirmation::ConfirmationHandler for DenyHandler {
            fn confirm(&self, _request: &ConfirmationRequest) -> ConfirmationResponse {
                ConfirmationResponse::Deny
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();
        config.require_plan_confirmation = true;

        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let final_answer = "Done.";
        let llm: Arc<dyn LlmService> = Arc::new(ScriptedLlm::new(vec![
            tool_call.into(),
            final_answer.into(),
        ]));
        let mut agent = AgentLoop::new(config, llm, make_registry(), make_services())
            .with_confirmation_handler(Arc::new(DenyHandler));

        let response = agent.run("q").unwrap();
        // 用户拒绝 -> Failed
        assert!(matches!(response.stop_reason, AgentStopReason::UserCancel));
    }

    #[test]
    fn confirmation_handler_approve_all_skips_future_confirmations() {
        use crate::confirmation::{ConfirmationRequest, ConfirmationResponse};

        struct ApproveAllHandler;
        impl crate::confirmation::ConfirmationHandler for ApproveAllHandler {
            fn confirm(&self, _request: &ConfirmationRequest) -> ConfirmationResponse {
                ConfirmationResponse::ApproveAll
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();
        config.require_plan_confirmation = true;

        let tool_call = r#"{"name":"list_files","arguments":{"path":"."}}"#;
        let final_answer = "Done.";
        let mut agent = make_agent(vec![tool_call.into(), final_answer.into()], config)
            .with_confirmation_handler(Arc::new(ApproveAllHandler));

        let response = agent.run("q").unwrap();
        // ApproveAll -> 执行继续 -> 完成
        assert_eq!(response.answer, "Done.");
        assert!(matches!(response.stop_reason, AgentStopReason::Completed));
    }
}
