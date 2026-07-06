//! `chatvcode agent` 子命令实现。
//!
//! 该模块将 [`chatvcode_agent`] 暴露的能力接入 CLI，支持：
//! - 单次调用模式：消费事件流并展示执行轨迹
//! - 交互式 REPL 模式：多轮对话 + 控制命令
//! - 简洁/详细/JSON 三种输出模式
//! - 统计信息（步数、工具调用数、token 使用、耗时）
//! - 友好的错误提示

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use chatvcode_agent::{
    AgentBuilder, AgentConfig, AgentEvent, AgentResponse, AgentServices, AgentStopReason,
    ChunkMetadataStoreAdapter, CodeSearchService, CoreSearchService, TokenBudgetConfig,
    ToolRetryConfig,
};
use chatvcode_core::{ChatVCodeError, ChatOptions, ErrorSeverity, ParseSource};
use chatvcode_llm::{
    ChatTemplate, GenerationParams, LlmConfig, LlmService, LlamaService, MockLlmService,
    auto_discover_model, estimate_memory, format_bytes, list_models,
};
use chatvcode_parser::parse_source;
use chatvcode_vdb::EmbeddingService;
use clap::Args;
use rustyline::DefaultEditor;

use crate::LlamaEmbeddingAdapter;

/// `chatvcode agent` 子命令参数。
#[derive(Args, Debug, Clone)]
pub struct AgentCommand {
    /// 用户问题（交互式模式下可省略，进入 REPL 后再问）。
    pub question: Option<String>,

    /// 待分析的项目路径（必需）。
    #[arg(short, long, default_value = ".", help = "Path to the project directory")]
    pub path: String,

    /// GGUF 模型路径（未指定时自动发现）。
    #[arg(short, long, help = "Path to the GGUF model file")]
    pub model: Option<String>,

    /// 最大步数（默认 10）。
    #[arg(long, help = "Maximum number of agent steps")]
    pub max_steps: Option<usize>,

    /// 超时秒数（默认 120）。
    #[arg(long, help = "Agent timeout in seconds")]
    pub timeout: Option<u64>,

    /// 详细模式：逐步打印思考、工具调用与结果。
    #[arg(short, long, default_value_t = false, num_args = 0..=1, help = "Verbose output")]
    pub verbose: bool,

    /// 限制可用工具，逗号分隔（默认全部 6 个内置工具）。
    #[arg(long, help = "Comma-separated list of allowed tools")]
    pub tools: Option<String>,

    /// 交互式多轮对话模式。
    #[arg(short, long, default_value_t = false, num_args = 0..=1, help = "Interactive REPL mode")]
    pub interactive: bool,

    /// 以 JSON 格式输出最终响应。
    #[arg(long, help = "Output response as JSON")]
    pub json: bool,

    /// 禁用流式输出，等执行完成后整体打印。
    #[arg(long, default_value_t = false, num_args = 0..=1, help = "Disable streaming output")]
    pub no_stream: bool,

    /// 启用规划确认：在执行前要求用户确认计划。
    #[arg(long, default_value_t = false, num_args = 0..=1, help = "Require plan confirmation")]
    pub confirm_plan: bool,

    /// 生成温度（默认 0.7）。
    #[arg(long, default_value = "0.7", help = "Temperature for generation")]
    pub temperature: f32,

    /// 最大生成 token 数（默认 2048）。
    #[arg(long, default_value = "2048", help = "Maximum tokens to generate")]
    pub max_tokens: i32,

    /// 上下文窗口大小（默认 8192）。
    #[arg(long, default_value = "8192", help = "Context window size")]
    pub n_ctx: u32,

    /// 线程数（默认自动）。
    #[arg(long, help = "Number of threads for inference")]
    pub n_threads: Option<i32>,

    /// GPU 层数（默认 0，-1 表示全部）。
    #[arg(long, default_value = "0", help = "Number of GPU layers (-1 for all)")]
    pub n_gpu_layers: i32,

    /// 嵌入模型路径（GGUF 或 ONNX）。若不指定，Agent 仍可使用文件级工具。
    #[arg(long, help = "Path to the embedding model file")]
    pub embedding_model: Option<String>,

    /// 使用 Mock LLM（测试用，无需真实模型）。
    #[arg(long, hide = true, help = "Use mock LLM service for testing")]
    pub mock_llm: bool,

    /// Mock LLM 回复内容。
    #[arg(long, hide = true, help = "Response text for mock LLM")]
    pub mock_llm_response: Option<String>,
}

/// 运行 `chatvcode agent` 命令。
pub fn run_agent(cmd: AgentCommand) -> Result<(), ChatVCodeError> {
    let project_path = PathBuf::from(&cmd.path);
    if !project_path.exists() {
        return Err(ChatVCodeError::invalid_input(format!(
            "Project path does not exist: {}",
            project_path.display()
        )));
    }

    let llm = setup_llm(&cmd)?;

    if cmd.interactive {
        return run_agent_interactive(cmd, llm);
    }

    let question = cmd.question.clone().ok_or_else(|| {
        ChatVCodeError::invalid_input(
            "Agent command requires a question argument, or use --interactive for REPL mode.",
        )
    })?;

    run_agent_single(&cmd, question, Arc::from(llm))
}

/// 运行一次 Agent 查询（单次模式）。
fn run_agent_single(
    cmd: &AgentCommand,
    question: String,
    llm: Arc<dyn LlmService>,
) -> Result<(), ChatVCodeError> {
    let config = build_agent_config(cmd);
    let services = build_agent_services(cmd)?;

    let (response, answer_printed): (AgentResponse, bool) = if cmd.no_stream {
        let agent = AgentBuilder::new(config)
            .with_llm(Arc::clone(&llm))
            .with_services(services);
        (agent.run(&question).map_err(agent_error_to_cli)?, false)
    } else {
        let rx = chatvcode_agent::agent_query_stream(
            &question,
            config,
            Arc::clone(&llm),
            Arc::clone(&services),
        )
        .map_err(agent_error_to_cli)?;
        let resp = consume_event_stream(cmd, rx)?;
        (resp, true)
    };

    if cmd.json {
        print_json(&response)?;
    } else {
        print_response(cmd, &response, answer_printed);
    }

    Ok(())
}

/// 消费事件流，并在 verbose 模式下实时展示执行轨迹。
fn consume_event_stream(
    cmd: &AgentCommand,
    rx: std::sync::mpsc::Receiver<AgentEvent>,
) -> Result<AgentResponse, ChatVCodeError> {
    let stdout = std::io::stdout();
    let verbose = cmd.verbose;

    loop {
        match rx.recv_timeout(std::time::Duration::from_secs(cmd.timeout.unwrap_or(120) + 30)) {
            Ok(event) => match event {
                AgentEvent::StateChanged { from, to } => {
                    if verbose {
                        eprintln!("[state] {from:?} -> {to:?}");
                    }
                }
                AgentEvent::ThinkingPhaseChanged { phase } => {
                    if verbose {
                        eprintln!("[phase] {phase:?}");
                    }
                }
                AgentEvent::Thinking { text } => {
                    if verbose {
                        eprintln!("[think] {text}");
                    }
                }
                AgentEvent::ToolCallStarted { name, arguments } => {
                    if verbose {
                        eprintln!("[tool] {name}({arguments})");
                    } else {
                        eprintln!("→ {name}");
                    }
                }
                AgentEvent::ToolCallCompleted { name, cached, .. } => {
                    if verbose {
                        eprintln!("[tool] {name} done (cached={cached})");
                    }
                }
                AgentEvent::ToolCallFailed { name, error, will_retry } => {
                    if verbose {
                        eprintln!("[tool] {name} failed: {error} (retry={will_retry})");
                    } else {
                        eprintln!("✗ {name} failed: {error}");
                    }
                }
                AgentEvent::ToolCallRetrying { name, attempt, max_attempts } => {
                    if verbose {
                        eprintln!("[tool] retry {name} {attempt}/{max_attempts}");
                    }
                }
                AgentEvent::StepCompleted { step } => {
                    if verbose {
                        eprintln!(
                            "[step {}] state={:?} tool_calls={} duration={}ms",
                            step.step_number,
                            step.state,
                            step.tool_calls.len(),
                            step.duration_ms
                        );
                    }
                }
                AgentEvent::LoopDetected { detection_type } => {
                    eprintln!("⚠ loop detected: {detection_type}");
                }
                AgentEvent::AnswerStarted => {
                    if !cmd.json {
                        eprintln!("\n--- Answer ---");
                    }
                }
                AgentEvent::AnswerToken { text } => {
                    if !cmd.json {
                        print!("{text}");
                        let _ = stdout.lock().flush();
                    }
                }
                AgentEvent::AnswerCompleted { response } => {
                    if !cmd.json {
                        eprintln!();
                    }
                    return Ok(response);
                }
                AgentEvent::Error { message } => {
                    return Err(ChatVCodeError::internal(message));
                }
                AgentEvent::TokenBudgetWarning { used, remaining } => {
                    if verbose {
                        eprintln!("[budget] used={used} remaining={remaining}");
                    }
                }
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(ChatVCodeError::internal("Agent stream timed out"));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(ChatVCodeError::internal(
                    "Agent stream disconnected before completion",
                ));
            }
        }
    }
}

/// 交互式 REPL 模式。
fn run_agent_interactive(
    cmd: AgentCommand,
    llm: Box<dyn LlmService>,
) -> Result<(), ChatVCodeError> {
    let llm: Arc<dyn LlmService> = Arc::from(llm);
    eprintln!();
    eprintln!("🤖 Agent interactive mode (type /quit to exit, /help for commands)");
    eprintln!("📂 Project: {}", cmd.path);
    eprintln!();

    let mut rl = DefaultEditor::new().map_err(|e| {
        ChatVCodeError::internal(format!("Failed to initialize line editor: {e}"))
    })?;
    if let Some(hp) = dirs::home_dir().map(|h| h.join(".chatvcode").join("agent_history")) {
        let _ = rl.load_history(&hp);
    }

    let mut verbose = cmd.verbose;
    let mut last_question: Option<String> = None;

    loop {
        let readline = rl.readline("🤖 > ");
        match readline {
            Ok(line) => {
                let input = line.trim().to_string();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(&input);

                if input.starts_with('/') {
                    match handle_agent_command(&input, &cmd, &mut verbose, &last_question) {
                        AgentReplAction::Continue => continue,
                        AgentReplAction::Quit => break,
                        AgentReplAction::Run(q) => {
                            last_question = Some(q.clone());
                            let mut single_cmd = cmd.clone();
                            single_cmd.verbose = verbose;
                            if let Err(e) = run_agent_single(&single_cmd, q, Arc::clone(&llm)) {
                                eprintln!("✗ {e}");
                            }
                        }
                    }
                } else {
                    last_question = Some(input.clone());
                    let mut single_cmd = cmd.clone();
                    single_cmd.verbose = verbose;
                    if let Err(e) = run_agent_single(&single_cmd, input, Arc::clone(&llm)) {
                        eprintln!("✗ {e}");
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                eprintln!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                eprintln!("👋 Goodbye!");
                break;
            }
            Err(e) => {
                eprintln!("✗ Input error: {e}");
                break;
            }
        }
    }

    if let Some(hp) = dirs::home_dir().map(|h| h.join(".chatvcode").join("agent_history")) {
        let _ = rl.save_history(&hp);
    }
    Ok(())
}

/// REPL 命令解析结果。
enum AgentReplAction {
    Continue,
    Quit,
    Run(String),
}

/// 处理 REPL 控制命令。
fn handle_agent_command(
    input: &str,
    cmd: &AgentCommand,
    verbose: &mut bool,
    last_question: &Option<String>,
) -> AgentReplAction {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let name = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).filter(|s| !s.is_empty());

    match name {
        "/help" | "/h" | "/?" => {
            print_agent_help();
            AgentReplAction::Continue
        }
        "/quit" | "/exit" | "/q" => {
            eprintln!("👋 Goodbye!");
            AgentReplAction::Quit
        }
        "/clear" => {
            eprintln!("✓ Agent history cleared (each turn is independent).");
            AgentReplAction::Continue
        }
        "/retry" | "/r" => {
            if let Some(q) = last_question {
                eprintln!("🔄 Retrying: {q}");
                AgentReplAction::Run(q.clone())
            } else {
                eprintln!("(No previous question to retry)");
                AgentReplAction::Continue
            }
        }
        "/continue" => {
            eprintln!("ℹ Continuing execution with relaxed step limit is not yet supported.");
            AgentReplAction::Continue
        }
        "/steps" => {
            eprintln!("ℹ Step history inspection in REPL mode is not yet supported.");
            AgentReplAction::Continue
        }
        "/tools" => {
            print_available_tools();
            AgentReplAction::Continue
        }
        "/verbose" => {
            *verbose = !*verbose;
            eprintln!("verbose: {}", if *verbose { "on" } else { "off" });
            AgentReplAction::Continue
        }
        "/budget" => {
            eprintln!("Token budget (configured): {:?}", build_agent_config(cmd).token_budget);
            AgentReplAction::Continue
        }
        "/export" => {
            let _ = arg;
            eprintln!("ℹ /export is not yet supported in agent REPL mode.");
            AgentReplAction::Continue
        }
        "/model" => {
            print_model_info(cmd);
            AgentReplAction::Continue
        }
        other => {
            eprintln!("Unknown command: {other}. Type /help for commands.");
            AgentReplAction::Continue
        }
    }
}

fn print_agent_help() {
    eprintln!("Agent REPL commands:");
    eprintln!("  /help, /h        Show this help");
    eprintln!("  /quit, /q        Exit");
    eprintln!("  /clear           Clear conversation state");
    eprintln!("  /retry, /r        Retry the last question");
    eprintln!("  /continue        Continue execution (not yet supported)");
    eprintln!("  /steps            Show step history (not yet supported)");
    eprintln!("  /tools           List available tools");
    eprintln!("  /verbose         Toggle verbose mode");
    eprintln!("  /budget          Show token budget");
    eprintln!("  /export <file>  Export session (not yet supported)");
    eprintln!("  /model           Show model info");
}

fn print_available_tools() {
    let tools = [
        "read_file", "list_files", "grep_code", "get_file_structure", "search_symbol",
        "search_code",
    ];
    eprintln!("Built-in tools:");
    for t in tools {
        eprintln!("  - {t}");
    }
}

fn print_model_info(cmd: &AgentCommand) {
    match &cmd.model {
        Some(p) => eprintln!("Model: {p}"),
        None => match auto_discover_model() {
            Ok(p) => eprintln!("Model (auto): {}", p.display()),
            Err(e) => eprintln!("Model auto-discovery failed: {e}"),
        },
    }
    if cmd.mock_llm {
        eprintln!("Mock LLM: enabled");
    }
}

/// 构造 LLM 服务。
fn setup_llm(cmd: &AgentCommand) -> Result<Box<dyn LlmService>, ChatVCodeError> {
    if cmd.mock_llm {
        let resp = cmd.mock_llm_response.clone().unwrap_or_else(|| {
            "Agent mock response: I would explore the codebase using the available tools."
                .to_string()
        });
        return Ok(Box::new(MockLlmService::new(resp)));
    }

    let model_path = match &cmd.model {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("🔍 Auto-discovering model...");
            let discovered = list_models();
            if let Some(first) = discovered.first() {
                eprintln!("✓ Found model: {}", first.path.display());
                first.path.clone()
            } else {
                match auto_discover_model() {
                    Ok(p) => p,
                    Err(e) => {
                        return Err(ChatVCodeError::internal(format!(
                            "Model auto-discovery failed: {e}. Specify --model=<path> or --mock-llm."
                        ))
                        .with_severity(ErrorSeverity::Unrecoverable));
                    }
                }
            }
        }
    };

    eprintln!("⏳ Loading model: {}...", model_path.display());
    if let Ok(estimate) = estimate_memory(&model_path, cmd.n_ctx) {
        eprintln!("  Memory estimate: {}", format_bytes(estimate.total_bytes));
    }

    let n_threads = cmd.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
    let config = LlmConfig::new(&model_path)
        .with_n_ctx(cmd.n_ctx)
        .with_n_threads(n_threads)
        .with_n_gpu_layers(cmd.n_gpu_layers);

    match LlamaService::new(&config) {
        Ok(svc) => {
            eprintln!("✓ Model loaded.");
            Ok(Box::new(svc) as Box<dyn LlmService>)
        }
        Err(e) => Err(ChatVCodeError::internal(format!("Failed to load model: {e}"))
            .with_severity(ErrorSeverity::Unrecoverable)),
    }
}

/// 构造 Agent 配置。
fn build_agent_config(cmd: &AgentCommand) -> AgentConfig {
    let mut config = AgentConfig::default();
    config.project_path = PathBuf::from(&cmd.path);
    config.verbose = cmd.verbose;
    config.require_plan_confirmation = cmd.confirm_plan;
    config.generation_params = GenerationParams::default()
        .with_temperature(cmd.temperature)
        .with_max_tokens(cmd.max_tokens);
    config.chat_template = ChatTemplate::Auto;
    config.token_budget = TokenBudgetConfig {
        total_budget: cmd.n_ctx as usize,
        ..TokenBudgetConfig::default()
    };
    config.tool_retry = ToolRetryConfig::default();

    if let Some(ms) = cmd.max_steps {
        config.max_steps = ms;
    }
    if let Some(t) = cmd.timeout {
        config.timeout_secs = t;
    }
    if let Some(tools_str) = &cmd.tools {
        config.allowed_tools = tools_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }
    config
}

/// 构造 AgentServices。
///
/// 若向量库 + 嵌入服务可用，使用 [`CoreSearchService`]；否则使用无检索的回退服务
/// （Agent 仍可通过 read_file / list_files / grep_code / get_file_structure 探索项目）。
fn build_agent_services(cmd: &AgentCommand) -> Result<Arc<AgentServices>, ChatVCodeError> {
    let project_path = PathBuf::from(&cmd.path);
    let parser: Box<dyn ParseSource> = Box::new(parse_source);

    let metadata_path = ChatOptions::new(&project_path).resolve_metadata_store_path();
    let metadata_store = chatvcode_core::model::ChunkMetadataStore::load_or_new(&metadata_path);
    let chunk_store = Box::new(ChunkMetadataStoreAdapter::new(metadata_store));

    let search: Box<dyn CodeSearchService> = match setup_embedding(cmd) {
        Ok(embedding) => {
            let parser_arc: Arc<dyn ParseSource> = Arc::new(parse_source);
            Box::new(CoreSearchService::new(project_path, parser_arc, embedding))
        }
        Err(e) => {
            if cmd.verbose {
                eprintln!("⚠ Embedding service unavailable: {e}");
                eprintln!("  Falling back to file-level tools only (search_code/search_symbol disabled).");
            }
            Box::new(NoSearchService)
        }
    };

    Ok(Arc::new(AgentServices {
        search,
        parser,
        chunk_store,
    }))
}

/// 设置嵌入服务（复用 chat 命令的 GGUF 适配）。
fn setup_embedding(cmd: &AgentCommand) -> Result<Box<dyn EmbeddingService>, ChatVCodeError> {
    let project_path = PathBuf::from(&cmd.path);
    let vector_store_path = ChatOptions::new(&project_path).resolve_vector_store_path();
    if !vector_store_path.exists() {
        return Err(ChatVCodeError::invalid_input(format!(
            "Vector store not found at {}. Run `chatvcode index {} --embedding-model=<MODEL>` first.",
            vector_store_path.display(),
            cmd.path
        )));
    }

    let model_path = match &cmd.embedding_model {
        Some(p) if p.to_lowercase().ends_with(".gguf") => PathBuf::from(p),
        Some(_) => {
            return Err(ChatVCodeError::invalid_input(
                "ONNX embedding model is not yet supported by the agent command. \
                 Use a GGUF embedding model via --embedding-model.",
            ));
        }
        None => match &cmd.model {
            Some(p) => PathBuf::from(p),
            None => match auto_discover_model() {
                Ok(p) => p,
                Err(e) => {
                    return Err(ChatVCodeError::internal(format!(
                        "Embedding model required for search_code tool: {e}"
                    )));
                }
            },
        },
    };

    let n_threads = cmd.n_threads.unwrap_or_else(|| num_cpus::get() as i32);
    let embed_svc = chatvcode_llm::LlamaEmbeddingService::from_path(
        &model_path,
        128,
        n_threads,
        0,
        false,
    )
    .map_err(|e| ChatVCodeError::internal(format!("Failed to load embedding model: {e}")))?;

    Ok(Box::new(LlamaEmbeddingAdapter::new(embed_svc)) as Box<dyn EmbeddingService>)
}

/// 无检索回退服务。
struct NoSearchService;

impl CodeSearchService for NoSearchService {
    fn search(
        &self,
        _query: &str,
        _top_k: usize,
    ) -> Result<Vec<chatvcode_core::model::SearchResult>, chatvcode_agent::AgentError> {
        Err(chatvcode_agent::AgentError::ToolError {
            tool_name: "search_code".into(),
            message: "No vector store available. Run `chatvcode index` first.".into(),
        })
    }
}

/// 打印简洁/详细模式的响应。
fn print_response(cmd: &AgentCommand, response: &AgentResponse, answer_printed: bool) {
    if !answer_printed {
        if !cmd.verbose {
            eprintln!("\n--- Answer ---");
        } else {
            eprintln!("\n--- Final Answer ---");
        }
        println!("{}", response.answer);
    }

    println!();
    print_statistics(response);
    print_stop_reason(&response.stop_reason);
}

/// 打印统计信息。
fn print_statistics(response: &AgentResponse) {
    eprintln!();
    eprintln!("📊 Statistics:");
    eprintln!("  Steps:         {}", response.metrics.total_steps);
    eprintln!("  Tool calls:    {}", response.total_tool_calls);
    eprintln!(
        "  Tokens:        {} prompt + {} completion = {} total",
        response.total_token_usage.prompt_tokens,
        response.total_token_usage.completion_tokens,
        response.total_token_usage.total_tokens
    );
    eprintln!("  Duration:      {:.2}s", response.total_duration_ms as f64 / 1000.0);
}

/// 打印停止原因。
fn print_stop_reason(reason: &AgentStopReason) {
    let label = match reason {
        AgentStopReason::Completed => "✓ Completed",
        AgentStopReason::MaxSteps => "⚠ Stopped: maximum steps reached",
        AgentStopReason::Timeout => "⚠ Stopped: timeout",
        AgentStopReason::UserCancel => "⚠ Stopped: user cancelled",
        AgentStopReason::LoopDetected => "⚠ Stopped: loop detected",
        AgentStopReason::Error(msg) => return eprintln!("✗ Error: {msg}"),
    };
    eprintln!("{label}");
}

/// JSON 模式输出。
fn print_json(response: &AgentResponse) -> Result<(), ChatVCodeError> {
    let json = serde_json::to_string_pretty(response)
        .map_err(|e| ChatVCodeError::internal(format!("Failed to serialize response: {e}")))?;
    println!("{json}");
    Ok(())
}

/// 将 AgentError 转换为 ChatVCodeError，并附带友好提示。
fn agent_error_to_cli(e: chatvcode_agent::AgentError) -> ChatVCodeError {
    let msg = e.to_string();
    match e {
        chatvcode_agent::AgentError::Timeout(_) => {
            ChatVCodeError::internal(format!("Agent timed out. Consider increasing --timeout. [{msg}]"))
                .with_severity(ErrorSeverity::Recoverable)
        }
        chatvcode_agent::AgentError::MaxStepsReached(_) => ChatVCodeError::internal(format!(
            "Agent reached the maximum step count. Consider increasing --max-steps. [{msg}]"
        ))
        .with_severity(ErrorSeverity::Recoverable),
        chatvcode_agent::AgentError::LlmError(m) => {
            ChatVCodeError::internal(format!("LLM error: {m}"))
                .with_severity(ErrorSeverity::Unrecoverable)
        }
        chatvcode_agent::AgentError::TokenBudgetExceeded { .. } => ChatVCodeError::internal(
            format!("Token budget exceeded. Increase --n-ctx or reduce context. [{msg}]"),
        )
        .with_severity(ErrorSeverity::Recoverable),
        other => ChatVCodeError::internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(subcommand)]
        cmd: TestCommand,
    }

    #[derive(clap::Subcommand)]
    enum TestCommand {
        Agent(AgentCommand),
    }

    #[test]
    fn agent_command_parses_question_and_path() {
        let cli = TestCli::try_parse_from([
            "test", "agent", "What does this project do?", "--path", "/tmp/proj",
        ]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert_eq!(agent.question.as_deref(), Some("What does this project do?"));
            assert_eq!(agent.path, "/tmp/proj");
        }
    }

    #[test]
    fn agent_command_default_path() {
        let cli = TestCli::try_parse_from(["test", "agent", "hello"]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert_eq!(agent.path, ".");
            assert!(!agent.verbose);
            assert!(!agent.interactive);
            assert!(!agent.json);
        }
    }

    #[test]
    fn agent_command_verbose_short_flag() {
        let cli = TestCli::try_parse_from(["test", "agent", "hello", "-v"]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert!(agent.verbose);
        }
    }

    #[test]
    fn agent_command_interactive_short_flag() {
        let cli = TestCli::try_parse_from(["test", "agent", "-i", "--path", "/tmp/p"]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert!(agent.interactive);
            assert_eq!(agent.path, "/tmp/p");
            assert!(agent.question.is_none());
        }
    }

    #[test]
    fn agent_command_json_and_no_stream() {
        let cli = TestCli::try_parse_from([
            "test", "agent", "hi", "--json", "--no-stream", "--max-steps", "5",
        ]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert!(agent.json);
            assert!(agent.no_stream);
            assert_eq!(agent.max_steps, Some(5));
        }
    }

    #[test]
    fn agent_command_tools_csv() {
        let cli = TestCli::try_parse_from([
            "test", "agent", "hi", "--tools", "read_file,grep_code",
        ]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert_eq!(agent.tools.as_deref(), Some("read_file,grep_code"));
        }
    }

    #[test]
    fn agent_command_mock_llm() {
        let cli = TestCli::try_parse_from([
            "test", "agent", "hi", "--mock-llm", "--mock-llm-response", "mocked",
        ]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert!(agent.mock_llm);
            assert_eq!(agent.mock_llm_response.as_deref(), Some("mocked"));
        }
    }

    #[test]
    fn agent_command_timeout_and_confirm_plan() {
        let cli = TestCli::try_parse_from([
            "test", "agent", "hi", "--timeout", "60", "--confirm-plan",
        ]);
        assert!(cli.is_ok());
        if let Ok(TestCli { cmd: TestCommand::Agent(agent) }) = cli {
            assert_eq!(agent.timeout, Some(60));
            assert!(agent.confirm_plan);
        }
    }

    #[test]
    fn build_agent_config_applies_cli_overrides() {
        let cmd = AgentCommand {
            question: Some("q".into()),
            path: "/p".into(),
            model: None,
            max_steps: Some(7),
            timeout: Some(30),
            verbose: true,
            tools: Some("read_file".into()),
            interactive: false,
            json: false,
            no_stream: false,
            confirm_plan: true,
            temperature: 0.5,
            max_tokens: 1024,
            n_ctx: 4096,
            n_threads: None,
            n_gpu_layers: 0,
            embedding_model: None,
            mock_llm: true,
            mock_llm_response: None,
        };
        let cfg = build_agent_config(&cmd);
        assert_eq!(cfg.max_steps, 7);
        assert_eq!(cfg.timeout_secs, 30);
        assert!(cfg.verbose);
        assert!(cfg.require_plan_confirmation);
        assert_eq!(cfg.allowed_tools, vec!["read_file"]);
        assert_eq!(cfg.token_budget.total_budget, 4096);
    }

    #[test]
    fn no_search_service_returns_error() {
        let svc = NoSearchService;
        let result = svc.search("anything", 5);
        assert!(result.is_err());
    }
}