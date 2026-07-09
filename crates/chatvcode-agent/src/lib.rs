pub mod agent_loop;
pub mod budget;
pub mod cache;
pub mod confirmation;
pub mod context;
pub mod error;
pub mod executor;
pub mod hot_reload;
pub mod loop_detector;
pub mod metrics;
pub mod orchestrator;
pub mod prompt;
pub mod prompt_optimizer;
pub mod scenario;
pub mod service;
pub mod session;
pub mod state;
pub mod tools;
pub mod trace;
pub mod types;

#[cfg(test)]
pub mod mock_llm;

pub use budget::{
    BudgetReport, SessionContext, SimpleTokenEstimator, TokenBudgetManager, TokenEstimator,
};
pub use agent_loop::AgentLoop;
pub use confirmation::{
    AlwaysDenyHandler, AutoApproveHandler, ConfirmationHandler, ConfirmationRequest,
    ConfirmationResponse,
};
pub use context::{
    AgentServices, ChunkMetadataStoreAdapter, ChunkMetadataStoreTrait, CodeSearchService,
    CoreSearchService, ToolContext,
};
pub use error::{AgentError, AgentResult};
pub use executor::{BuiltinToolRegistry, ToolExecutor};
pub use hot_reload::RuntimeToolRegistry;
pub use loop_detector::{LoopDetectionResult, LoopDetector};
pub use orchestrator::{
    AgentMessage, AgentOrchestrator, AgentTask, DefaultAgentRunner, OrchestrationResult,
    SubAgentRunner,
};
pub use prompt::AgentPromptBuilder;
pub use prompt_optimizer::{
    OptimizationResult, PromptMetric, PromptOptimizer, PromptRunRunner, PromptVariant, VariantScore,
};
pub use scenario::{
    DocumentScenario, GenericTextScenario, Scenario, ScenarioRegistry, ListDocsTool,
    ReadTextFileTool, SearchTextTool,
};
pub use service::{AgentBuilder, AgentService, agent_query, agent_query_stream};
pub use session::AgentSession;
pub use state::{AgentStateMachine, TransitionEvent};
pub use trace::{PerformanceBenchmark, TraceRenderer};
pub use tools::{
    BuiltinTool, CompareFilesTool, EditFileTool, FindReferencesTool, GetDependenciesTool,
    GetFileStructureTool, GetProjectOverviewTool, GrepCodeTool, ListFilesTool, ReadFileTool,
    RunCommandTool, SearchCodeTool, SearchSymbolTool, WriteFileTool,
};
pub use types::*;
