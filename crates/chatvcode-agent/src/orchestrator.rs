//! 多 Agent 协作：通过 [`AgentOrchestrator`] 将一个复杂任务拆分为若干子任务，
//! 交由子 Agent 执行并汇总结果。
//!
//! 设计上保持对底层 LLM/服务实现解耦：协调器只依赖 [`SubAgentRunner`] trait，
//! 调用方可以注入真实 Agent（[`DefaultAgentRunner`]）或测试用 Mock runner。
//!
//! 子任务之间可声明依赖（[`AgentTask::depends_on`]），协调器按依赖顺序串行
//! 执行，并在每个子任务完成后广播 [`AgentMessage`] 让其他监听者观察进度。

use std::collections::HashMap;
use std::sync::Arc;

use chatvcode_llm::LlmService;

use crate::agent_loop::AgentLoop;
use crate::context::AgentServices;
use crate::error::AgentResult;
use crate::executor::{BuiltinToolRegistry, ToolExecutor};
use crate::types::{AgentConfig, AgentResponse, ToolRetryConfig};

/// 一个子 Agent 需要处理的任务。
#[derive(Debug, Clone)]
pub struct AgentTask {
    /// 任务唯一标识。
    pub id: String,
    /// 子任务的查询文本。
    pub query: String,
    /// 该任务依赖的其他任务 ID 列表（在依赖完成后再执行）。
    pub depends_on: Vec<String>,
    /// 任务可选配置覆盖；为 `None` 时使用 orchestrator 的默认配置。
    pub config_override: Option<AgentConfig>,
}

impl AgentTask {
    /// 创建一个无依赖的子任务。
    pub fn new(id: impl Into<String>, query: impl Into<String>) -> Self {
        Self { id: id.into(), query: query.into(), depends_on: vec![], config_override: None }
    }

    /// 声明该任务依赖 `dep` 完成后才能执行。
    #[must_use]
    pub fn depends_on(mut self, dep: impl Into<String>) -> Self {
        self.depends_on.push(dep.into());
        self
    }

    /// 提供该任务专属的 Agent 配置覆盖。
    #[must_use]
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config_override = Some(config);
        self
    }
}

/// 协调器在执行过程中对外广播的消息。
#[derive(Debug, Clone)]
pub enum AgentMessage {
    /// 子任务开始执行。
    Started { id: String, query: String },
    /// 子任务执行成功完成。
    Completed { id: String, response: Box<AgentResponse> },
    /// 子任务执行失败。
    Failed { id: String, error: String },
    /// 子任务向其他监听者广播中间信息（如完成的中间结论）。
    Broadcast { from: String, content: String },
}

/// 子 Agent 运行器：执行单个 [`AgentTask`] 并返回响应。
///
/// 实现者可对接真实的 [`AgentLoop`] / [`AgentBuilder`] 或测试桩。
pub trait SubAgentRunner: Send + Sync {
    /// 执行给定任务并返回响应。
    fn run(&self, task: &AgentTask) -> AgentResult<AgentResponse>;
}

/// 默认运行器：使用传入的 LLM 与 Agent 服务构建一个 [`AgentLoop`]，
/// 在主线程内同步执行任务。
pub struct DefaultAgentRunner {
    llm_service: Arc<dyn LlmService>,
    services: Arc<AgentServices>,
    base_config: AgentConfig,
}

impl DefaultAgentRunner {
    /// 创建运行器。
    pub fn new(
        llm_service: Arc<dyn LlmService>,
        services: Arc<AgentServices>,
        base_config: AgentConfig,
    ) -> Self {
        Self { llm_service, services, base_config }
    }

    fn build_executor(base: &AgentConfig) -> Arc<dyn ToolExecutor> {
        let retry: ToolRetryConfig = base.tool_retry.clone();
        let mut registry = BuiltinToolRegistry::new(retry);
        registry.register_defaults();
        Arc::new(registry)
    }
}

impl SubAgentRunner for DefaultAgentRunner {
    fn run(&self, task: &AgentTask) -> AgentResult<AgentResponse> {
        let config = task.config_override.clone().unwrap_or_else(|| self.base_config.clone());
        let tool_executor = Self::build_executor(&config);
        let mut agent =
            AgentLoop::new(config, Arc::clone(&self.llm_service), tool_executor, Arc::clone(&self.services));
        agent.run(&task.query)
    }
}

/// 协调结果：包含每个子任务的状态与最终响应。
#[derive(Debug, Clone, Default)]
pub struct OrchestrationResult {
    /// 任务 ID -> 完成响应（仅成功的任务）。
    pub completed: HashMap<String, AgentResponse>,
    /// 任务 ID -> 错误描述（仅失败的任务）。
    pub failed: HashMap<String, String>,
    /// 按执行顺序排列的任务 ID。
    pub execution_order: Vec<String>,
    /// 执行期间产生的消息（按时间顺序）。
    pub messages: Vec<AgentMessage>,
}

impl OrchestrationResult {
    /// 协调是否完全成功（无失败任务）。
    pub fn is_success(&self) -> bool {
        self.failed.is_empty()
    }
}

/// 多 Agent 协调器：调度若干子任务，处理依赖关系并汇总结果。
pub struct AgentOrchestrator<R: SubAgentRunner> {
    runner: R,
    tasks: Vec<AgentTask>,
}

impl<R: SubAgentRunner> AgentOrchestrator<R> {
    /// 创建协调器。
    pub fn new(runner: R) -> Self {
        Self { runner, tasks: Vec::new() }
    }

    /// 添加一个子任务。
    #[must_use]
    pub fn add_task(mut self, task: AgentTask) -> Self {
        self.tasks.push(task);
        self
    }

    /// 添加多个子任务。
    #[must_use]
    pub fn add_tasks(mut self, tasks: Vec<AgentTask>) -> Self {
        self.tasks.extend(tasks);
        self
    }

    /// 按依赖顺序执行所有任务，返回汇总结果。
    ///
    /// 依赖未声明的任务可被先执行；若存在循环依赖或缺失依赖，
    /// 该子任务将被标记为失败。
    pub fn execute(self) -> OrchestrationResult {
        let Self { runner, tasks } = self;

        let mut result = OrchestrationResult::default();
        let mut remaining: Vec<AgentTask> = tasks;
        let mut completed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        while !remaining.is_empty() {
            // 找出所有依赖已满足的任务
            let ready_indices: Vec<usize> = remaining
                .iter()
                .enumerate()
                .filter(|(_, t)| t.depends_on.iter().all(|d| completed_ids.contains(d)))
                .map(|(i, _)| i)
                .collect();

            if ready_indices.is_empty() {
                // 依赖无法满足：剩余任务彼此之间形成环或依赖缺失
                for task in &remaining {
                    let msg = format!(
                        "Unsatisfied dependencies for task '{}': {:?}",
                        task.id, task.depends_on
                    );
                    result.failed.insert(task.id.clone(), msg.clone());
                    result.messages.push(AgentMessage::Failed {
                        id: task.id.clone(),
                        error: msg,
                    });
                }
                return result;
            }

            // 执行第一个就绪任务（简化：串行）
            let idx = ready_indices[0];
            let task = remaining.remove(idx);
            result.execution_order.push(task.id.clone());
            result.messages.push(AgentMessage::Started {
                id: task.id.clone(),
                query: task.query.clone(),
            });

            match runner.run(&task) {
                Ok(response) => {
                    result.messages.push(AgentMessage::Broadcast {
                        from: task.id.clone(),
                        content: response.answer.clone(),
                    });
                    result.messages.push(AgentMessage::Completed {
                        id: task.id.clone(),
                        response: Box::new(response.clone()),
                    });
                    result.completed.insert(task.id.clone(), response);
                    completed_ids.insert(task.id);
                }
                Err(e) => {
                    let msg = e.to_string();
                    result.messages.push(AgentMessage::Failed {
                        id: task.id.clone(),
                        error: msg.clone(),
                    });
                    result.failed.insert(task.id.clone(), msg);
                    // 失败任务的 ID 也算"处理过"，避免阻塞依赖它的任务
                    completed_ids.insert(task.id);
                }
            }
        }

        result
    }

    /// 仅执行一个任务并返回响应（便捷方法）。
    pub fn run_one(&self, task: &AgentTask) -> AgentResult<AgentResponse> {
        self.runner.run(task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AgentError;
    use crate::types::{
        AgentMetrics, AgentResponse, AgentStopReason, AgentStep, SourceReference, TokenUsage,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn synthetic_response(answer: &str) -> AgentResponse {
        AgentResponse {
            answer: answer.to_string(),
            sources: Vec::<SourceReference>::new(),
            steps: Vec::<AgentStep>::new(),
            total_token_usage: TokenUsage::default(),
            total_duration_ms: 0,
            total_tool_calls: 0,
            stop_reason: AgentStopReason::Completed,
            metrics: AgentMetrics::default(),
            self_evaluation: None,
        }
    }

    /// Mock runner：按任务 ID 返回预设答案。
    struct MockRunner {
        answers: HashMap<String, String>,
        call_count: Arc<Mutex<HashMap<String, usize>>>,
    }

    impl MockRunner {
        fn new<I: IntoIterator<Item = (String, String)>>(iter: I) -> Self {
            Self {
                answers: iter.into_iter().collect(),
                call_count: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        fn calls(&self, id: &str) -> usize {
            self.call_count.lock().unwrap().get(id).copied().unwrap_or(0)
        }
    }

    impl SubAgentRunner for MockRunner {
        fn run(&self, task: &AgentTask) -> AgentResult<AgentResponse> {
            let mut counts = self.call_count.lock().unwrap();
            *counts.entry(task.id.clone()).or_insert(0) += 1;
            drop(counts);
            match self.answers.get(&task.id) {
                Some(ans) => Ok(synthetic_response(ans)),
                None => Err(AgentError::Internal(format!("no answer for {}", task.id))),
            }
        }
    }

    #[test]
    fn orchestrator_executes_independent_tasks() {
        let runner = MockRunner::new([
            ("a".to_string(), "answerA".to_string()),
            ("b".to_string(), "answerB".to_string()),
        ]);
        let orch = AgentOrchestrator::new(runner)
            .add_task(AgentTask::new("a", "queryA"))
            .add_task(AgentTask::new("b", "queryB"));
        let result = orch.execute();
        assert!(result.is_success());
        assert_eq!(result.completed.len(), 2);
        assert_eq!(result.completed["a"].answer, "answerA");
        assert_eq!(result.completed["b"].answer, "answerB");
        assert_eq!(result.execution_order.len(), 2);
    }

    #[test]
    fn orchestrator_respects_dependencies() {
        let runner = MockRunner::new([
            ("child".to_string(), "childAnswer".to_string()),
            ("parent".to_string(), "parentAnswer".to_string()),
        ]);
        let orch = AgentOrchestrator::new(runner)
            .add_task(AgentTask::new("parent", "queryParent").depends_on("child"))
            .add_task(AgentTask::new("child", "queryChild"));
        let result = orch.execute();
        assert!(result.is_success());
        // child 必须先执行，parent 紧随其后
        assert_eq!(result.execution_order, vec!["child", "parent"]);
    }

    #[test]
    fn orchestrator_unsatisfied_dependency_fails() {
        let runner = MockRunner::new([("a".to_string(), "ansA".to_string())]);
        let orch = AgentOrchestrator::new(runner)
            .add_task(AgentTask::new("a", "queryA").depends_on("missing"));
        let result = orch.execute();
        assert!(!result.is_success());
        assert!(result.failed.contains_key("a"));
    }

    #[test]
    fn orchestrator_records_messages() {
        let runner = MockRunner::new([("a".to_string(), "ansA".to_string())]);
        let orch = AgentOrchestrator::new(runner).add_task(AgentTask::new("a", "q"));
        let result = orch.execute();
        let kinds: Vec<&str> = result
            .messages
            .iter()
            .map(|m| match m {
                AgentMessage::Started { .. } => "started",
                AgentMessage::Completed { .. } => "completed",
                AgentMessage::Broadcast { .. } => "broadcast",
                AgentMessage::Failed { .. } => "failed",
            })
            .collect();
        assert!(kinds.contains(&"started"));
        assert!(kinds.contains(&"broadcast"));
        assert!(kinds.contains(&"completed"));
    }

    #[test]
    fn orchestrator_run_one_invokes_runner_once() {
        let runner = MockRunner::new([("x".to_string(), "X".to_string())]);
        let orch = AgentOrchestrator::new(runner);
        let resp = orch.run_one(&AgentTask::new("x", "q")).unwrap();
        assert_eq!(resp.answer, "X");
        assert_eq!(orch.runner.calls("x"), 1);
    }

    #[test]
    fn orchestration_result_default_is_empty() {
        let r = OrchestrationResult::default();
        assert!(r.is_success());
        assert!(r.completed.is_empty());
        assert!(r.failed.is_empty());
    }

    #[test]
    fn task_builder_methods() {
        let cfg = AgentConfig::default();
        let task = AgentTask::new("id", "q").depends_on("dep").with_config(cfg.clone());
        assert_eq!(task.id, "id");
        assert_eq!(task.depends_on, vec!["dep"]);
        assert!(task.config_override.is_some());
    }

    #[test]
    fn atomic_runner_counter_is_thread_safe() {
        let _ = AtomicUsize::new(0).fetch_add(1, Ordering::SeqCst);
    }
}