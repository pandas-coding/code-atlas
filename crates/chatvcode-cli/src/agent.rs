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
    ChunkMetadataStoreAdapter, CodeSearchService, CoreSearchService, PerformanceBenchmark,
    TokenBudgetConfig, ToolRetryConfig, TraceRenderer,
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

    /// 在同一步骤中并行执行多个工具调用。
    #[arg(long, default_value_t = false, num_args = 0..=1, help = "Enable parallel tool execution")]
    pub parallel_tools: bool,

    /// 在最终回答后执行自我评估。
    #[arg(long, default_value_t = false, num_args = 0..=1, help = "Enable self-evaluation after answering")]
    pub self_eval: bool,

    /// 将执行轨迹导出为 HTML（.html）或 Markdown（.md）文件。
    #[arg(long, help = "Export execution trace to file (format by extension: .html or .md)")]
    pub trace: Option<String>,
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

    run_agent_single(&cmd, question, Arc::from(llm)).map(|_| ())
}

/// 运行一次 Agent 查询（单次模式）。
fn run_agent_single(
    cmd: &AgentCommand,
    question: String,
    llm: Arc<dyn LlmService>,
) -> Result<AgentResponse, ChatVCodeError> {
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

    // 若指定了 --trace，导出执行轨迹到文件
    if let Some(trace_path) = &cmd.trace {
        let path = PathBuf::from(trace_path);
        let content = if path.extension().and_then(|e| e.to_str()) == Some("html") {
            TraceRenderer::render_html(&response)
        } else {
            TraceRenderer::render_markdown(&response)
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match std::fs::write(&path, content) {
            Ok(()) => eprintln!("✓ Trace exported to {}", path.display()),
            Err(e) => eprintln!("✗ Failed to write trace '{}': {e}", path.display()),
        }
    }

    Ok(response)
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
    let mut last_response: Option<AgentResponse> = None;
    let mut benchmark = PerformanceBenchmark::new();

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
                    match handle_agent_command(
                        &input,
                        &cmd,
                        &mut verbose,
                        &last_question,
                        &mut last_response,
                        &mut benchmark,
                    ) {
                        AgentReplAction::Continue => continue,
                        AgentReplAction::Quit => break,
                        AgentReplAction::Run(q) => {
                            last_question = Some(q.clone());
                            let mut single_cmd = cmd.clone();
                            single_cmd.verbose = verbose;
                            match run_agent_single(&single_cmd, q, Arc::clone(&llm)) {
                                Ok(resp) => {
                                    benchmark.add(&resp);
                                    last_response = Some(resp);
                                }
                                Err(e) => eprintln!("✗ {e}"),
                            }
                        }
                    }
                } else {
                    last_question = Some(input.clone());
                    let mut single_cmd = cmd.clone();
                    single_cmd.verbose = verbose;
                    match run_agent_single(&single_cmd, input, Arc::clone(&llm)) {
                        Ok(resp) => {
                            benchmark.add(&resp);
                            last_response = Some(resp);
                        }
                        Err(e) => eprintln!("✗ {e}"),
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

/// 默认的 Agent 会话保存路径（`~/.chatvcode/agent_session.json`）。
fn agent_default_session_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".chatvcode").join("agent_session.json"))
}

/// 处理 REPL 控制命令。
fn handle_agent_command(
    input: &str,
    cmd: &AgentCommand,
    verbose: &mut bool,
    last_question: &Option<String>,
    last_response: &mut Option<AgentResponse>,
    benchmark: &mut PerformanceBenchmark,
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
            *last_response = None;
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
            if let Some(resp) = last_response.as_ref() {
                print_session_steps(resp);
            } else {
                eprintln!("(No previous session. Run a query first.)");
            }
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
        "/export" | "/save" => {
            handle_save_command(arg, last_response);
            AgentReplAction::Continue
        }
        "/load" => {
            handle_load_command(arg, last_response);
            AgentReplAction::Continue
        }
        "/trace" => {
            handle_trace_command(arg, last_response);
            AgentReplAction::Continue
        }
        "/benchmark" | "/bench" => {
            handle_benchmark_command(benchmark);
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

/// 处理 `/save [path]` 命令：将最近一次 Agent 响应序列化为 JSON 并写入文件。
fn handle_save_command(arg: Option<&str>, last_response: &Option<AgentResponse>) {
    let Some(resp) = last_response.as_ref() else {
        eprintln!("(No session to save. Run a query first.)");
        return;
    };

    let path = match arg.map(PathBuf::from) {
        Some(p) => p,
        None => match agent_default_session_path() {
            Some(p) => p,
            None => {
                eprintln!("✗ Could not determine default save path. Specify a path: /save <path>");
                return;
            }
        },
    };

    match serde_json::to_string_pretty(resp) {
        Ok(json) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("✗ Failed to create directory '{}': {e}", parent.display());
                        return;
                    }
                }
            }
            match std::fs::write(&path, json) {
                Ok(()) => eprintln!("✓ Session saved to {}", path.display()),
                Err(e) => eprintln!("✗ Failed to write '{}': {e}", path.display()),
            }
        }
        Err(e) => eprintln!("✗ Failed to serialize session: {e}"),
    }
}

/// 处理 `/load [path]` 命令：从 JSON 文件恢复 Agent 响应并展示摘要。
fn handle_load_command(arg: Option<&str>, last_response: &mut Option<AgentResponse>) {
    let path = match arg.map(PathBuf::from) {
        Some(p) => p,
        None => match agent_default_session_path() {
            Some(p) => p,
            None => {
                eprintln!("✗ Could not determine default load path. Specify a path: /load <path>");
                return;
            }
        },
    };

    if !path.exists() {
        eprintln!("✗ File not found: {}", path.display());
        return;
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("✗ Failed to read '{}': {e}", path.display());
            return;
        }
    };

    match serde_json::from_str::<AgentResponse>(&content) {
        Ok(resp) => {
            eprintln!("✓ Session loaded from {}", path.display());
            print_session_summary(&resp);
            *last_response = Some(resp);
        }
        Err(e) => {
            eprintln!("✗ Failed to parse session JSON: {e}");
        }
    }
}

/// 打印会话步骤摘要（用于 `/steps` 命令）。
fn print_session_steps(resp: &AgentResponse) {
    if resp.steps.is_empty() {
        eprintln!("(No steps recorded.)");
        return;
    }
    eprintln!("📋 Session steps ({} total):", resp.steps.len());
    for step in &resp.steps {
        let tool_count = step.tool_calls.len();
        eprintln!(
            "  Step {}: {:?} ({} tool call{}, {}ms, {} tokens)",
            step.step_number,
            step.state,
            tool_count,
            if tool_count == 1 { "" } else { "s" },
            step.duration_ms,
            step.token_usage.total_tokens,
        );
        for call in &step.tool_calls {
            eprintln!("    → {}({})", call.name, call.arguments.len());
        }
    }
}

/// 打印已加载会话的摘要（用于 `/load` 命令）。
fn print_session_summary(resp: &AgentResponse) {
    eprintln!("📊 Session summary:");
    eprintln!("  Steps:         {}", resp.metrics.total_steps);
    eprintln!("  Tool calls:    {}", resp.total_tool_calls);
    eprintln!(
        "  Tokens:        {} prompt + {} completion = {} total",
        resp.total_token_usage.prompt_tokens,
        resp.total_token_usage.completion_tokens,
        resp.total_token_usage.total_tokens,
    );
    eprintln!("  Duration:      {:.2}s", resp.total_duration_ms as f64 / 1000.0);
    eprintln!("  Stop reason:   {:?}", resp.stop_reason);
    if !resp.answer.is_empty() {
        eprintln!("  Answer:        {}", truncate_str(&resp.answer, 200));
    }
}

/// 处理 `/trace [path]` 命令：将最近一次执行的轨迹导出为 HTML 或 Markdown。
///
/// 文件扩展名决定格式：`.html` -> HTML，`.md` -> Markdown。
/// 不指定路径时输出 Markdown 到 stderr。
fn handle_trace_command(arg: Option<&str>, last_response: &Option<AgentResponse>) {
    let Some(resp) = last_response.as_ref() else {
        eprintln!("(No session to trace. Run a query first.)");
        return;
    };

    match arg {
        Some(path_str) => {
            let path = PathBuf::from(path_str);
            let content = if path.extension().and_then(|e| e.to_str()) == Some("html") {
                TraceRenderer::render_html(resp)
            } else {
                TraceRenderer::render_markdown(resp)
            };

            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("✗ Failed to create directory '{}': {e}", parent.display());
                        return;
                    }
                }
            }
            match std::fs::write(&path, content) {
                Ok(()) => eprintln!("✓ Trace exported to {}", path.display()),
                Err(e) => eprintln!("✗ Failed to write '{}': {e}", path.display()),
            }
        }
        None => {
            let md = TraceRenderer::render_markdown(resp);
            eprintln!("{md}");
        }
    }
}

/// 处理 `/benchmark` 命令：输出聚合的性能基准报告。
fn handle_benchmark_command(benchmark: &PerformanceBenchmark) {
    if benchmark.run_count == 0 {
        eprintln!("(No runs recorded. Run queries first.)");
        return;
    }
    let md = benchmark.render_markdown();
    eprintln!("{md}");
}

/// 将字符串截断到指定字符数并附加省略号。
fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let prefix: String = s.chars().take(max).collect();
    format!("{prefix}...")
}

fn print_agent_help() {
    eprintln!("Agent REPL commands:");
    eprintln!("  /help, /h         Show this help");
    eprintln!("  /quit, /q         Exit");
    eprintln!("  /clear            Clear conversation state");
    eprintln!("  /retry, /r        Retry the last question");
    eprintln!("  /continue         Continue execution (not yet supported)");
    eprintln!("  /steps            Show step history of the last session");
    eprintln!("  /tools            List available tools");
    eprintln!("  /verbose          Toggle verbose mode");
    eprintln!("  /budget           Show token budget");
    eprintln!("  /save [path]      Save last session to JSON (default: ~/.chatvcode/agent_session.json)");
    eprintln!("  /load [path]      Load session from JSON (default: ~/.chatvcode/agent_session.json)");
    eprintln!("  /trace [path]     Export execution trace as HTML (.html) or Markdown (.md)");
    eprintln!("  /benchmark        Show aggregated performance benchmark");
    eprintln!("  /export [path]    Alias of /save");
    eprintln!("  /model            Show model info");
}

fn print_available_tools() {
    let tools = [
        "read_file",
        "list_files",
        "grep_code",
        "get_file_structure",
        "search_symbol",
        "search_code",
        "find_references",
        "get_dependencies",
        "compare_files",
        "get_project_overview",
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
    config.enable_parallel_tool_calls = cmd.parallel_tools;
    config.enable_self_evaluation = cmd.self_eval;
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
            parallel_tools: true,
            self_eval: true,
            trace: None,
        };
        let cfg = build_agent_config(&cmd);
        assert_eq!(cfg.max_steps, 7);
        assert_eq!(cfg.timeout_secs, 30);
        assert!(cfg.verbose);
        assert!(cfg.require_plan_confirmation);
        assert!(cfg.enable_parallel_tool_calls);
        assert!(cfg.enable_self_evaluation);
        assert_eq!(cfg.allowed_tools, vec!["read_file"]);
        assert_eq!(cfg.token_budget.total_budget, 4096);
    }

    #[test]
    fn no_search_service_returns_error() {
        let svc = NoSearchService;
        let result = svc.search("anything", 5);
        assert!(result.is_err());
    }

    #[test]
    fn save_command_writes_json_file() {
        use chatvcode_agent::{AgentMetrics, AgentResponse, AgentStopReason, SourceReference,
            TokenUsage};
        
        

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("session.json");

        let resp = AgentResponse {
            answer: "test answer".into(),
            sources: vec![SourceReference {
                file_path: "src/main.rs".into(),
                line_start: 1,
                line_end: 5,
                symbol_name: Some("main".into()),
                relevance: 1.0,
            }],
            steps: vec![],
            total_token_usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            total_duration_ms: 100,
            total_tool_calls: 0,
            stop_reason: AgentStopReason::Completed,
            metrics: AgentMetrics::default(),
            self_evaluation: None,
        };

        handle_save_command(Some(path.to_str().unwrap()), &Some(resp));

        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("test answer"));
        assert!(content.contains("src/main.rs"));
    }

    #[test]
    fn save_command_without_session_prints_message() {
        handle_save_command(Some("/nonexistent/save.json"), &None);
    }

    #[test]
    fn load_command_reads_json_file() {
        use chatvcode_agent::{AgentMetrics, AgentResponse};
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("loaded.json");

        let json = serde_json::json!({
            "answer": "loaded answer",
            "sources": [],
            "steps": [],
            "total_token_usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "total_duration_ms": 50,
            "total_tool_calls": 1,
            "stop_reason": "Completed",
            "metrics": AgentMetrics::default(),
        });
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(serde_json::to_string(&json).unwrap().as_bytes())
            .unwrap();

        let mut last: Option<AgentResponse> = None;
        handle_load_command(Some(path.to_str().unwrap()), &mut last);

        assert!(last.is_some());
        assert_eq!(last.unwrap().answer, "loaded answer");
    }

    #[test]
    fn load_command_missing_file_prints_error() {
        let mut last: Option<AgentResponse> = None;
        handle_load_command(Some("/nonexistent/load.json"), &mut last);
        assert!(last.is_none());
    }

    #[test]
    fn truncate_str_short_and_long() {
        assert_eq!(truncate_str("hi", 10), "hi");
        let long = "x".repeat(50);
        let t = truncate_str(&long, 10);
        assert!(t.ends_with("..."));
        assert_eq!(t.chars().count(), 13);
    }
}