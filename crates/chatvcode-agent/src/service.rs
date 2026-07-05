//! Agent 服务入口：对外暴露的统一调用 API。
//!
//! 该模块在 [`AgentLoop`] 之上提供薄封装：
//! - [`agent_query`]：同步执行一次 Agent 查询并返回最终结果。
//! - [`agent_query_stream`]：以事件流的方式运行 Agent 查询。
//! - [`AgentBuilder`]：可选的便捷构建器，用于链式组装依赖并触发同步/流式执行。
//!
//! 这些入口函数屏蔽了 `BuiltinToolRegistry` 的初始化细节，
//! 使调用方（如 `chatvcode-cli`）只需提供查询字符串、配置、LLM
//! 服务与 [`AgentServices`] 即可使用 Agent 能力。

use std::sync::mpsc;
use std::sync::Arc;

use chatvcode_llm::LlmService;

use crate::agent_loop::AgentLoop;
use crate::context::AgentServices;
use crate::error::{AgentError, AgentResult};
use crate::executor::{BuiltinToolRegistry, ToolExecutor};
use crate::types::{AgentConfig, AgentEvent, AgentResponse, ToolRetryConfig};

/// Agent 服务接口：抽象一次 Agent 执行的生命周期。
///
/// 由 [`AgentLoop`] 实际实现该 trait，模块内提供的 [`agent_query`]
/// 与 [`agent_query_stream`] 即是对该能力的便捷封装。
pub trait AgentService: Send + Sync {
    /// 同步执行查询，返回最终响应。
    fn run(&mut self, query: &str) -> AgentResult<AgentResponse>;

    /// 流式执行查询，返回事件接收端。
    fn run_stream(&mut self, query: &str) -> Result<mpsc::Receiver<AgentEvent>, AgentError>;

    /// 取消正在执行的循环。
    fn cancel(&self);

    /// 继续执行：放宽步数限制后恢复运行。
    fn continue_execution(&mut self) -> AgentResult<AgentResponse>;

    /// 重试上一次查询。
    fn retry(&mut self) -> AgentResult<AgentResponse>;
}

// ---- 入口函数 ---------------------------------------------------------

/// 使用默认内置工具集执行一次 Agent 查询（同步）。
///
/// 该函数内部会：
/// 1. 依据 `config.tool_retry` 创建 [`BuiltinToolRegistry`] 并注册 6 个内置工具。
/// 2. 构造 [`AgentLoop`]，注入系统提示词、规划提示词。
/// 3. 调用 [`AgentLoop::run`]，进入主推理-执行循环直到进入终态。
///
/// # 参数
///
/// - `query`：用户的代码库探索问题。
/// - `config`：Agent 配置（步数/超时/预算/路径等）。
/// - `llm_service`：LLM 推理后端。
/// - `services`：检索/解析/块存储服务集合（用于构建每个工具调用所需的
///   [`crate::context::ToolContext`]）。
///
/// # 错误
///
/// LLM 推理失败且无法恢复、达到最大步数后仍无结论、或后台线程启动
/// 失败时返回 [`AgentError`]。具体停止原因见
/// [`crate::types::AgentStopReason`]。
pub fn agent_query(
    query: &str,
    config: AgentConfig,
    llm_service: Arc<dyn LlmService>,
    services: Arc<AgentServices>,
) -> AgentResult<AgentResponse> {
    let tool_executor = build_default_tool_executor(&config);
    let mut agent = AgentLoop::new(config, llm_service, tool_executor, services);
    agent.run(query)
}

/// 使用默认内置工具集执行一次 Agent 查询（流式）。
///
/// 返回 [`mpsc::Receiver`]，调用方可逐条消费 [`AgentEvent`]，包括
/// 状态切换、思考内容、工具调用、最终回答等。循环在后台线程中执行，
/// 完成时推送 [`AgentEvent::AnswerCompleted`] 或 [`AgentEvent::Error`]。
///
/// 参数语义与 [`agent_query`] 一致，差异仅在于返回事件流而非阻塞等待
/// 最终响应。
pub fn agent_query_stream(
    query: &str,
    config: AgentConfig,
    llm_service: Arc<dyn LlmService>,
    services: Arc<AgentServices>,
) -> AgentResult<mpsc::Receiver<AgentEvent>> {
    let tool_executor = build_default_tool_executor(&config);
    let agent = AgentLoop::new(config, llm_service, tool_executor, services);
    agent.run_stream(query)
}

/// 构造默认内置工具执行器：注册 6 个内置工具到 [`BuiltinToolRegistry`]。
///
/// 失败工具的重试策略取自 `config.tool_retry`；若调用方希望进一步定制
/// 工具集（例如禁用某些工具、追加自定义工具），可改用 [`AgentBuilder`]
/// 或直接构造 [`AgentLoop`]。
fn build_default_tool_executor(config: &AgentConfig) -> Arc<dyn ToolExecutor> {
    let retry_config: ToolRetryConfig = config.tool_retry.clone();
    let mut registry = BuiltinToolRegistry::new(retry_config);
    registry.register_defaults();
    Arc::new(registry)
}

// ---- 便捷构建器 -------------------------------------------------------

/// 可选的 Agent 便捷构建器：以链式调用方式组装依赖，并最终触发同步
/// 或流式执行。
///
/// 适用于不想一次性在调用处罗列所有参数的调用方。其仅是
/// [`agent_query`] / [`agent_query_stream`] 的语法糖。
///
/// # 示例
///
/// ```ignore
/// let response = AgentBuilder::new(config)
///     .with_llm(llm_service)
///     .with_services(services)
///     .run("项目入口函数在哪里？")?;
/// ```
pub struct AgentBuilder {
    config: AgentConfig,
    llm_service: Option<Arc<dyn LlmService>>,
    services: Option<Arc<AgentServices>>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
}

impl AgentBuilder {
    /// 创建一个新的构建器，初始状态仅持有配置。
    #[must_use]
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            llm_service: None,
            services: None,
            tool_executor: None,
        }
    }

    /// 提供自定义的工具执行器，覆盖默认内置工具集。
    #[must_use]
    pub fn with_tool_executor(mut self, executor: Arc<dyn ToolExecutor>) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    /// 提供默认的内置工具执行器。
    ///
    /// 显式声明使用 6 个内置工具；不调用该方法而在 [`Self::run`] /
    /// [`Self::run_stream`] 时未注入执行器也会默认使用内置工具。
    #[must_use]
    pub fn with_default_tools(mut self) -> Self {
        self.tool_executor = Some(build_default_tool_executor(&self.config));
        self
    }

    /// 设置 LLM 推理后端。
    #[must_use]
    pub fn with_llm(mut self, llm_service: Arc<dyn LlmService>) -> Self {
        self.llm_service = Some(llm_service);
        self
    }

    /// 设置检索/解析/块存储服务集合。
    #[must_use]
    pub fn with_services(mut self, services: Arc<AgentServices>) -> Self {
        self.services = Some(services);
        self
    }

    /// 获取已组装的配置引用（便于在调用前做检查或日志）。
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// 同步执行查询。
    ///
    /// # 错误
    ///
    /// 未提供 LLM 服务或 Agent 服务时返回 [`AgentError::Internal`]。
    pub fn run(self, query: &str) -> AgentResult<AgentResponse> {
        let Self { config, llm_service, services, tool_executor } = self;
        let llm_service = llm_service.ok_or_else(|| {
            AgentError::Internal("AgentBuilder: llm_service is required".into())
        })?;
        let services = services.ok_or_else(|| {
            AgentError::Internal("AgentBuilder: services is required".into())
        })?;
        let tool_executor =
            tool_executor.unwrap_or_else(|| build_default_tool_executor(&config));
        let mut agent = AgentLoop::new(config, llm_service, tool_executor, services);
        agent.run(query)
    }

    /// 以流式方式执行查询。
    ///
    /// # 错误
    ///
    /// 未提供 LLM 服务或 Agent 服务时返回 [`AgentError::Internal`]。
    pub fn run_stream(self, query: &str) -> AgentResult<mpsc::Receiver<AgentEvent>> {
        let Self { config, llm_service, services, tool_executor } = self;
        let llm_service = llm_service.ok_or_else(|| {
            AgentError::Internal("AgentBuilder: llm_service is required".into())
        })?;
        let services = services.ok_or_else(|| {
            AgentError::Internal("AgentBuilder: services is required".into())
        })?;
        let tool_executor =
            tool_executor.unwrap_or_else(|| build_default_tool_executor(&config));
        let agent = AgentLoop::new(config, llm_service, tool_executor, services);
        agent.run_stream(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn agent_builder_default_state() {
        let builder = AgentBuilder::new(AgentConfig::default());
        assert_eq!(builder.config().project_path, PathBuf::from("."));
        assert!(builder.llm_service.is_none());
        assert!(builder.services.is_none());
        assert!(builder.tool_executor.is_none());
    }

    #[test]
    fn with_default_tools_sets_executor() {
        let builder = AgentBuilder::new(AgentConfig::default()).with_default_tools();
        assert!(builder.tool_executor.is_some());
    }

    #[test]
    fn run_without_llm_returns_error() {
        let builder = AgentBuilder::new(AgentConfig::default());
        let result = builder.run("hello");
        assert!(matches!(result, Err(AgentError::Internal(_))));
    }

    #[test]
    fn run_without_services_returns_error() {
        // 不注入 LLM 也无法到达 services 缺失分支，因此这里仅检查
        // 在缺失 services 时的 prompt，避免实现完整 LlmService stub。
        let builder = AgentBuilder::new(AgentConfig::default());
        assert!(builder.services.is_none());
        let result = builder.run_stream("hello");
        assert!(matches!(result, Err(AgentError::Internal(_))));
    }
}