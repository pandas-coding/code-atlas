//! Agent 测试专用的可编程 Mock LLM 服务。
//!
//! 该模块仅在 `#[cfg(test)]` 下编译，提供：
//! - [`MockLlmResponse`]：描述单次 LLM 响应的枚举（最终回答 / 工具调用 /
//!   混合内容 / 格式错误 / 错误）。
//! - [`MockAgentLlm`]：按调用顺序依次返回预设响应的 [`LlmService`] 实现，
//!   用于驱动 [`AgentLoop`](crate::agent_loop::AgentLoop) 的多步推理。
//! - [`MockLlmResponse`] 上的工厂方法（[`direct_answer`](MockLlmResponse::direct_answer)、
//!   [`single_tool_call`](MockLlmResponse::single_tool_call)、
//!   [`multi_step`](MockLlmResponse::multi_step)），便于在测试中快速构造常见场景。
//!
//! 这些工具让 Agent 循环测试不依赖真实模型文件，可在 CI 中稳定运行。

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use chatvcode_llm::{
    GenerationParams, InferenceResponse, LlmError, LlmResult, LlmService, ModelInfo, StopReason,
    StreamEvent, TokenUsage, ToolCall,
};

/// 一次可编程的 Mock LLM 响应。
///
/// 每个变体对应 Agent 循环中可能遇到的一类 LLM 输出：
/// - [`Answer`](Self::Answer)：纯文本最终回答，不含工具调用。
/// - [`ToolCalls`](Self::ToolCalls)：一个或多个工具调用，序列化为 JSON。
/// - [`Mixed`](Self::Mixed)：文本与工具调用混合（如 `<think>...</think>` + JSON）。
/// - [`Malformed`](Self::Malformed)：无法解析为工具调用的原始文本，退化为最终回答。
/// - [`Error`](Self::Error)：模拟推理失败，返回 [`LlmError`]。
#[derive(Debug, Clone)]
pub enum MockLlmResponse {
    /// 纯文本最终回答。
    Answer(String),
    /// 工具调用列表，会被序列化为可被 [`parse_tool_calls`](chatvcode_llm::parse_tool_calls) 解析的 JSON。
    ToolCalls(Vec<ToolCall>),
    /// 思考文本与工具调用混合：先输出 `thought`，再追加工具调用的 JSON。
    Mixed { thought: String, tool_calls: Vec<ToolCall> },
    /// 原始文本（不会被解析为工具调用），用作格式错误或纯文本场景。
    Malformed(String),
    /// 模拟推理错误。
    Error(String),
}

impl MockLlmResponse {
    /// 构造一个直接回答场景。
    pub fn direct_answer(text: impl Into<String>) -> Self {
        Self::Answer(text.into())
    }

    /// 构造一个单工具调用场景。
    pub fn single_tool_call(name: impl Into<String>, arguments: HashMap<String, serde_json::Value>) -> Self {
        Self::ToolCalls(vec![make_tool_call(name, arguments)])
    }

    /// 构造一个多工具调用场景（同一步骤中并行调用多个工具）。
    pub fn multi_step(calls: Vec<ToolCall>) -> Self {
        Self::ToolCalls(calls)
    }

    /// 将该响应渲染为 LLM 输出文本。
    ///
    /// - [`Answer`](Self::Answer) / [`Malformed`](Self::Malformed)：原样返回。
    /// - [`ToolCalls`](Self::ToolCalls)：单个调用渲染为 JSON 对象，多个渲染为 JSON 数组。
    /// - [`Mixed`](Self::Mixed)：思考文本 + 工具调用 JSON。
    /// - [`Error`](Self::Error)：返回空字符串（实际不会到达，调用方会先返回错误）。
    pub fn render(&self) -> String {
        match self {
            Self::Answer(s) | Self::Malformed(s) => s.clone(),
            Self::ToolCalls(calls) => render_tool_calls(calls),
            Self::Mixed { thought, tool_calls } => {
                if tool_calls.is_empty() {
                    return thought.clone();
                }
                format!("{thought}\n{}", render_tool_calls(tool_calls))
            }
            Self::Error(_) => String::new(),
        }
    }
}

/// 将工具调用列表渲染为可被 [`parse_tool_calls`](chatvcode_llm::parse_tool_calls) 解析的 JSON 文本。
///
/// 单个调用渲染为对象 `{"name":...,"arguments":{...}}`，多个调用渲染为数组。
pub fn render_tool_calls(calls: &[ToolCall]) -> String {
    let values: Vec<serde_json::Value> = calls.iter().map(tool_call_to_json).collect();
    if values.len() == 1 {
        serde_json::to_string(&values[0]).unwrap_or_default()
    } else {
        serde_json::to_string(&values).unwrap_or_default()
    }
}

/// 将单个 [`ToolCall`] 渲染为 JSON 对象（包含可选 `id`）。
pub fn tool_call_to_json(call: &ToolCall) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), serde_json::Value::String(call.name.clone()));
    obj.insert(
        "arguments".into(),
        serde_json::Value::Object(
            call.arguments
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
    );
    if let Some(id) = &call.id {
        obj.insert("id".into(), serde_json::Value::String(id.clone()));
    }
    serde_json::Value::Object(obj)
}

/// 可编程的 Mock LLM 服务：按调用顺序依次返回预设的 [`MockLlmResponse`]。
///
/// 超出预设脚本后返回一个兜底的纯文本回答，避免测试因脚本耗尽而 panic。
pub struct MockAgentLlm {
    script: std::sync::Mutex<Vec<MockLlmResponse>>,
    call_count: AtomicUsize,
}

impl MockAgentLlm {
    /// 创建一个按 `script` 顺序响应的 Mock 服务。
    pub fn new(script: Vec<MockLlmResponse>) -> Self {
        Self { script: std::sync::Mutex::new(script), call_count: AtomicUsize::new(0) }
    }

    /// [`new`](Self::new) 的别名，与 design.md 中的命名保持一致。
    #[must_use]
    pub fn with_responses(responses: Vec<MockLlmResponse>) -> Self {
        Self::new(responses)
    }

    /// 预设场景：单步直接回答。
    #[must_use]
    pub fn direct_answer(answer: impl Into<String>) -> Self {
        Self::new(vec![MockLlmResponse::Answer(answer.into())])
    }

    /// 预设场景：单步工具调用 + 最终回答。
    #[must_use]
    pub fn single_tool_call(
        tool: impl Into<String>,
        arguments: HashMap<String, serde_json::Value>,
        answer: impl Into<String>,
    ) -> Self {
        Self::new(vec![
            MockLlmResponse::single_tool_call(tool, arguments),
            MockLlmResponse::Answer(answer.into()),
        ])
    }

    /// 预设场景：多步工具调用。
    ///
    /// `steps` 中每个元素是一组同步骤并行的工具调用；若附带思考文本，
    /// 则渲染为 [`MockLlmResponse::Mixed`]。
    #[must_use]
    pub fn multi_step(steps: Vec<(Vec<ToolCall>, Option<String>)>) -> Self {
        let mut responses = Vec::with_capacity(steps.len());
        for (calls, thought) in steps {
            match thought {
                Some(t) => responses.push(MockLlmResponse::Mixed { thought: t, tool_calls: calls }),
                None => responses.push(MockLlmResponse::ToolCalls(calls)),
            }
        }
        Self::new(responses)
    }

    /// 已被消费的响应数（便于测试断言）。
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// 取出下一个预设响应；脚本耗尽时返回兜底回答。
    ///
    /// 使用 `clone` 而非 `remove`，避免索引随移位变化导致后续取值错位。
    fn next_response(&self) -> MockLlmResponse {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        let script = self.script.lock().expect("script mutex poisoned");
        if count < script.len() {
            script[count].clone()
        } else {
            MockLlmResponse::Answer("I have no further information.".into())
        }
    }
}

impl LlmService for MockAgentLlm {
    fn infer(
        &self,
        _prompt: &str,
        _params: &GenerationParams,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<InferenceResponse> {
        if let Some(flag) = cancel_flag
            && flag.load(Ordering::Relaxed)
        {
            return Ok(InferenceResponse {
                text: String::new(),
                stop_reason: StopReason::Cancelled,
                token_usage: TokenUsage::new(0, 0),
                duration: Duration::from_millis(0),
                time_to_first_token: None,
                tokens_per_second: 0.0,
            });
        }

        let response = self.next_response();
        match response {
            MockLlmResponse::Error(msg) => Err(LlmError::Internal(msg)),
            other => {
                let text = other.render();
                let completion_tokens = (text.len() / 4).max(1) as i32;
                Ok(InferenceResponse {
                    text,
                    stop_reason: StopReason::Eos,
                    token_usage: TokenUsage::new(10, completion_tokens),
                    duration: Duration::from_millis(1),
                    time_to_first_token: Some(Duration::from_millis(1)),
                    tokens_per_second: 100.0,
                })
            }
        }
    }

    fn infer_stream(
        &self,
        _prompt: &str,
        _params: &GenerationParams,
        _cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<mpsc::Receiver<StreamEvent>> {
        Err(LlmError::Internal("MockAgentLlm does not support streaming".into()))
    }

    fn model_info(&self) -> LlmResult<ModelInfo> {
        Ok(ModelInfo {
            description: "MockAgentLlm".into(),
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

/// 构造一个 [`ToolCall`]，便于在测试脚本中组装参数。
pub fn make_tool_call(name: impl Into<String>, arguments: HashMap<String, serde_json::Value>) -> ToolCall {
    ToolCall { name: name.into(), arguments, id: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_services() -> Arc<crate::context::AgentServices> {
        use crate::context::{ChunkMetadataStoreTrait, CodeSearchService};
        use chatvcode_core::model::{ChunkMetadata, SearchResult};

        struct EmptySearch;
        impl CodeSearchService for EmptySearch {
            fn search(&self, _q: &str, _k: usize) -> Result<Vec<SearchResult>, crate::error::AgentError> {
                Ok(vec![])
            }
        }
        struct EmptyStore;
        impl ChunkMetadataStoreTrait for EmptyStore {
            fn get_chunks_by_symbol(&self, _s: &str, _k: Option<&str>) -> Vec<ChunkMetadata> {
                vec![]
            }
            fn get_chunk_by_id(&self, _id: &str) -> Option<ChunkMetadata> {
                None
            }
        }
        Arc::new(crate::context::AgentServices {
            search: Box::new(EmptySearch),
            parser: Box::new(|_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<
                chatvcode_core::model::ParseResult,
            > { unimplemented!() }),
            chunk_store: Box::new(EmptyStore),
        })
    }

    fn make_registry() -> Arc<dyn crate::executor::ToolExecutor> {
        let mut reg = crate::executor::BuiltinToolRegistry::new(crate::types::ToolRetryConfig::default());
        reg.register_defaults();
        Arc::new(reg)
    }

    fn make_config(max_steps: usize) -> crate::types::AgentConfig {
        crate::types::AgentConfig {
            max_steps,
            timeout_secs: 0,
            ..crate::types::AgentConfig::default()
        }
    }

    fn make_agent(script: Vec<MockLlmResponse>, config: crate::types::AgentConfig) -> crate::agent_loop::AgentLoop {
        let llm: Arc<dyn LlmService> = Arc::new(MockAgentLlm::new(script));
        crate::agent_loop::AgentLoop::new(config, llm, make_registry(), make_services())
    }

    fn call(name: &str, args: &[(&str, serde_json::Value)]) -> ToolCall {
        let mut map = HashMap::new();
        for (k, v) in args {
            map.insert((*k).to_string(), v.clone());
        }
        make_tool_call(name, map)
    }

    // ---- 14-1: Mock 基础能力 ------------------------------------------------

    #[test]
    fn render_single_tool_call_is_parseable() {
        let c = call("list_files", &[("path", serde_json::json!("."))]);
        let text = render_tool_calls(std::slice::from_ref(&c));
        let parsed = chatvcode_llm::parse_tool_calls(&text).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "list_files");
    }

    #[test]
    fn render_multiple_tool_calls_is_parseable() {
        let calls = vec![
            call("list_files", &[("path", serde_json::json!("."))]),
            call("read_file", &[("path", serde_json::json!("a.rs"))]),
        ];
        let text = render_tool_calls(&calls);
        let parsed = chatvcode_llm::parse_tool_calls(&text).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "list_files");
        assert_eq!(parsed[1].name, "read_file");
    }

    #[test]
    fn mock_agent_llm_consumes_script_in_order() {
        let llm = MockAgentLlm::new(vec![
            MockLlmResponse::direct_answer("first"),
            MockLlmResponse::direct_answer("second"),
        ]);
        let params = GenerationParams::default();
        let r1 = llm.infer("", &params, None).unwrap();
        let r2 = llm.infer("", &params, None).unwrap();
        let r3 = llm.infer("", &params, None).unwrap();
        assert_eq!(r1.text, "first");
        assert_eq!(r2.text, "second");
        assert_eq!(r3.text, "I have no further information.");
        assert_eq!(llm.call_count(), 3);
    }

    #[test]
    fn mock_agent_llm_returns_error_variant() {
        let llm = MockAgentLlm::new(vec![MockLlmResponse::Error("boom".into())]);
        let params = GenerationParams::default();
        let result = llm.infer("", &params, None);
        assert!(result.is_err());
    }

    #[test]
    fn factory_methods_build_expected_variants() {
        match MockLlmResponse::direct_answer("hi") {
            MockLlmResponse::Answer(s) => assert_eq!(s, "hi"),
            _ => panic!("expected Answer"),
        }
        match MockLlmResponse::single_tool_call("list_files", HashMap::new()) {
            MockLlmResponse::ToolCalls(c) => assert_eq!(c.len(), 1),
            _ => panic!("expected ToolCalls"),
        }
        match MockLlmResponse::multi_step(vec![
            call("list_files", &[("path", serde_json::json!("."))]),
            call("read_file", &[("path", serde_json::json!("a.rs"))]),
        ]) {
            MockLlmResponse::ToolCalls(c) => assert_eq!(c.len(), 2),
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn mock_agent_llm_constructors_match_design() {
        // 与 design.md 一致：direct_answer / single_tool_call / multi_step / with_responses
        let llm = MockAgentLlm::direct_answer("hello");
        let params = GenerationParams::default();
        assert_eq!(llm.infer("", &params, None).unwrap().text, "hello");

        let mut args = HashMap::new();
        args.insert("path".into(), serde_json::json!("."));
        let llm = MockAgentLlm::single_tool_call("list_files", args, "done");
        let first = llm.infer("", &params, None).unwrap().text;
        assert!(first.contains("list_files"), "first response should be the tool call");
        let second = llm.infer("", &params, None).unwrap().text;
        assert_eq!(second, "done");
        assert_eq!(llm.call_count(), 2);

        let steps = vec![(vec![call("list_files", &[("path", serde_json::json!("."))])], Some("thinking".into()))];
        let llm = MockAgentLlm::multi_step(steps);
        let first = llm.infer("", &params, None).unwrap().text;
        assert!(first.contains("thinking"));
        assert!(first.contains("list_files"));

        let _llm2 = MockAgentLlm::with_responses(vec![MockLlmResponse::Answer("a".into())]);
    }

    // ---- 14-2: LLM 异常行为测试 ---------------------------------------------

    #[test]
    fn malformed_json_falls_back_to_final_answer() {
        // 格式错误的 JSON 无法解析为工具调用 -> 视为最终回答
        let mut agent = make_agent(
            vec![MockLlmResponse::Malformed("{not valid json".into())],
            make_config(5),
        );
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "{not valid json");
        assert!(matches!(response.stop_reason, crate::types::AgentStopReason::Completed));
    }

    #[test]
    fn calling_nonexistent_tool_recovers_and_answers() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();

        let bad = call("nonexistent_tool", &[]);
        let final_ans = MockLlmResponse::direct_answer("Could not use the tool.");
        let mut agent = make_agent(vec![MockLlmResponse::ToolCalls(vec![bad]), final_ans], config);
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "Could not use the tool.");
        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == crate::types::AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert!(!acting[0].tool_results[0].success);
    }

    #[test]
    fn missing_required_argument_produces_failed_tool_result() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();

        // read_file 需要 path 参数，此处省略 -> 参数校验失败
        let bad = call("read_file", &[]);
        let final_ans = MockLlmResponse::direct_answer("Recovered from missing arg.");
        let mut agent = make_agent(vec![MockLlmResponse::ToolCalls(vec![bad]), final_ans], config);
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "Recovered from missing arg.");
        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == crate::types::AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert!(!acting[0].tool_results[0].success);
    }

    #[test]
    fn mixed_text_and_tool_calls_executes_tool() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();

        // 文本 + 工具调用混合：parse_tool_calls 的回退分支应从文本中提取 JSON
        let mixed = MockLlmResponse::Mixed {
            thought: "Let me list the files first.".into(),
            tool_calls: vec![call("list_files", &[("path", serde_json::json!("."))])],
        };
        let final_ans = MockLlmResponse::direct_answer("Done listing files.");
        let mut agent = make_agent(vec![mixed, final_ans], config);
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "Done listing files.");
        assert_eq!(response.total_tool_calls, 1);
    }

    #[test]
    fn think_tags_followed_by_tool_call_still_parses() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        let mut config = make_config(10);
        config.project_path = tmp.path().to_path_buf();

        // 模型输出 <think>reasoning</think> 后再给出工具调用 JSON。
        // parse_tool_calls 的"从文本中提取 JSON 对象"回退分支应能识别。
        let raw = format!(
            "<think>I should list files.</think>\n{}",
            render_tool_calls(std::slice::from_ref(&call(
                "list_files",
                &[("path", serde_json::json!("."))]
            )))
        );
        let final_ans = MockLlmResponse::direct_answer("Listed.");
        let mut agent = make_agent(
            vec![MockLlmResponse::Malformed(raw), final_ans],
            config,
        );
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "Listed.");
        assert_eq!(response.total_tool_calls, 1);
    }

    #[test]
    fn think_tags_without_tool_call_becomes_final_answer() {
        // 仅含 <think> 标签、无工具调用 JSON -> 视为最终回答
        let raw = "<think>reasoning only</think>".to_string();
        let mut agent = make_agent(vec![MockLlmResponse::Malformed(raw)], make_config(5));
        let response = agent.run("q").unwrap();
        assert_eq!(response.answer, "<think>reasoning only</think>");
        assert!(matches!(response.stop_reason, crate::types::AgentStopReason::Completed));
    }

    #[test]
    fn llm_error_propagates_to_agent_failure() {
        let mut agent = make_agent(
            vec![MockLlmResponse::Error("inference failed".into())],
            make_config(5),
        );
        let result = agent.run("q");
        assert!(result.is_err());
    }
}

/// 端到端测试：Mock LLM + 真实索引 + 真实工具 + 临时项目。
#[cfg(test)]
mod e2e_tests {
    use super::*;
    use crate::agent_loop::AgentLoop;
    use crate::context::{AgentServices, ChunkMetadataStoreAdapter, CoreSearchService};
    use crate::executor::{BuiltinToolRegistry, ToolExecutor};
    use crate::types::{AgentConfig, AgentState, AgentStopReason, ToolRetryConfig};
    use chatvcode_core::{IndexOptions, index_path_with_options, model::ChunkMetadataStore};
    use chatvcode_vdb::{EmbeddingService, EmbeddingVector, InMemoryVectorStore, VectorStore};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// 创建一个包含若干 Rust 源文件的临时项目。
    fn create_test_project() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/main.rs"),
            "fn main() {\n    println!(\"hello world\");\n}",
        )
        .unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn greet(name: &str) -> String {\n    format!(\"Hello, {}!\", name)\n}",
        )
        .unwrap();
        std::fs::write(
            root.join("src/utils.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
        )
        .unwrap();
        tmp
    }

    /// 用真实 parser 索引项目，并保存元数据/向量库到 `.chatvcode/`。
    fn setup_index(tmp: &TempDir) {
        let parser = chatvcode_parser::parse_source;
        let index_result =
            index_path_with_options(tmp.path(), &parser, &IndexOptions::default()).unwrap();
        assert!(index_result.stats.total_chunks > 0, "should have chunks");

        let chatvcode_dir = tmp.path().join(".chatvcode");
        std::fs::create_dir_all(&chatvcode_dir).unwrap();

        let metadata_store = chatvcode_core::build_metadata_store(&index_result);
        let metadata_path = chatvcode_dir.join("vectors.atmd");
        metadata_store.save(&metadata_path).unwrap();

        let embedding_service = chatvcode_vdb::MockEmbeddingService::new(32);
        let all_chunks: Vec<_> = index_result
            .files
            .iter()
            .flat_map(|f| f.chunks.iter())
            .collect();
        let texts: Vec<&str> = all_chunks.iter().map(|c| c.source_text.as_str()).collect();
        let vectors = embedding_service.embed(&texts).unwrap();
        let evs: Vec<EmbeddingVector> = all_chunks
            .iter()
            .zip(vectors)
            .map(|(chunk, vector)| EmbeddingVector::new(&chunk.id, vector))
            .collect();
        let mut store = InMemoryVectorStore::new();
        store.add(evs).unwrap();
        store.save(&chatvcode_dir.join("vectors.db")).unwrap();
    }

    /// 构造真实 AgentServices：CoreSearchService + ChunkMetadataStoreAdapter + 真实 parser。
    fn build_real_services(project_path: PathBuf) -> Arc<AgentServices> {
        let metadata_path = project_path.join(".chatvcode").join("vectors.atmd");
        let metadata_store = ChunkMetadataStore::load_or_new(&metadata_path);
        let chunk_store = Box::new(ChunkMetadataStoreAdapter::new(metadata_store));

        let embedding: Box<dyn chatvcode_vdb::EmbeddingService> =
            Box::new(chatvcode_vdb::MockEmbeddingService::new(32));
        let parser_arc: Arc<dyn chatvcode_core::ParseSource> = Arc::new(chatvcode_parser::parse_source);
        let search = Box::new(CoreSearchService::new(project_path, parser_arc, embedding));

        let parser: Box<dyn chatvcode_core::ParseSource> = Box::new(chatvcode_parser::parse_source);
        Arc::new(AgentServices { search, parser, chunk_store })
    }

    fn make_registry() -> Arc<dyn ToolExecutor> {
        let mut reg = BuiltinToolRegistry::new(ToolRetryConfig::default());
        reg.register_defaults();
        Arc::new(reg)
    }

    fn make_config(project_path: PathBuf, max_steps: usize) -> AgentConfig {
        AgentConfig {
            max_steps,
            timeout_secs: 0,
            project_path,
            ..AgentConfig::default()
        }
    }

    fn call(name: &str, args: &[(&str, serde_json::Value)]) -> ToolCall {
        let mut map = HashMap::new();
        for (k, v) in args {
            map.insert((*k).to_string(), v.clone());
        }
        make_tool_call(name, map)
    }

    #[test]
    fn e2e_search_code_returns_results_and_final_answer() {
        let tmp = create_test_project();
        setup_index(&tmp);

        let project_path = tmp.path().to_path_buf();
        let services = build_real_services(project_path.clone());
        let registry = make_registry();

        let script = vec![
            MockLlmResponse::ToolCalls(vec![call(
                "search_code",
                &[("query", serde_json::json!("greet function")), ("top_k", serde_json::json!(3))],
            )]),
            MockLlmResponse::direct_answer("The greet function returns a greeting string."),
        ];
        let llm: Arc<dyn LlmService> = Arc::new(MockAgentLlm::new(script));

        let mut agent = AgentLoop::new(make_config(project_path, 10), llm, registry, services);
        let response = agent.run("What does greet do?").unwrap();

        assert_eq!(response.answer, "The greet function returns a greeting string.");
        assert!(matches!(response.stop_reason, AgentStopReason::Completed));
        assert_eq!(response.total_tool_calls, 1);

        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert_eq!(acting[0].tool_calls[0].name, "search_code");
        assert!(acting[0].tool_results[0].success, "search_code should succeed");
    }

    #[test]
    fn e2e_file_level_tools_explore_project_without_index() {
        // 未建立索引：search_code 不可用（无向量库），但 list_files / read_file 仍可工作。
        let tmp = create_test_project();
        let project_path = tmp.path().to_path_buf();
        let services = build_real_services(project_path.clone());
        let registry = make_registry();

        let script = vec![
            MockLlmResponse::ToolCalls(vec![call(
                "list_files",
                &[("path", serde_json::json!("."))],
            )]),
            MockLlmResponse::ToolCalls(vec![call(
                "read_file",
                &[("path", serde_json::json!("src/utils.rs"))],
            )]),
            MockLlmResponse::direct_answer("The add function adds two integers."),
        ];
        let llm: Arc<dyn LlmService> = Arc::new(MockAgentLlm::new(script));

        let mut agent = AgentLoop::new(make_config(project_path, 10), llm, registry, services);
        let response = agent.run("What does add do?").unwrap();

        assert_eq!(response.answer, "The add function adds two integers.");
        assert!(matches!(response.stop_reason, AgentStopReason::Completed));
        assert_eq!(response.total_tool_calls, 2);

        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 2);
        assert_eq!(acting[0].tool_calls[0].name, "list_files");
        assert_eq!(acting[1].tool_calls[0].name, "read_file");
        // read_file 结果应包含源码片段
        let read_result = &acting[1].tool_results[0];
        assert!(read_result.success);
        let content = serde_json::to_string(&read_result.value).unwrap();
        assert!(content.contains("a + b"), "read_file result should contain source text");
    }

    #[test]
    fn e2e_search_symbol_uses_real_metadata_store() {
        let tmp = create_test_project();
        setup_index(&tmp);

        let project_path = tmp.path().to_path_buf();
        let services = build_real_services(project_path.clone());
        let registry = make_registry();

        let script = vec![
            MockLlmResponse::ToolCalls(vec![call(
                "search_symbol",
                &[("symbol", serde_json::json!("greet")), ("kind", serde_json::json!("function"))],
            )]),
            MockLlmResponse::direct_answer("Found greet in lib.rs."),
        ];
        let llm: Arc<dyn LlmService> = Arc::new(MockAgentLlm::new(script));

        let mut agent = AgentLoop::new(make_config(project_path, 10), llm, registry, services);
        let response = agent.run("Where is greet?").unwrap();

        assert_eq!(response.answer, "Found greet in lib.rs.");
        assert_eq!(response.total_tool_calls, 1);

        let acting: Vec<_> = response
            .steps
            .iter()
            .filter(|s| s.state == AgentState::Acting)
            .collect();
        assert_eq!(acting.len(), 1);
        assert_eq!(acting[0].tool_calls[0].name, "search_symbol");
        assert!(acting[0].tool_results[0].success);
        let content = serde_json::to_string(&acting[0].tool_results[0].value).unwrap();
        assert!(content.contains("greet"), "search_symbol result should mention greet");
    }
}
