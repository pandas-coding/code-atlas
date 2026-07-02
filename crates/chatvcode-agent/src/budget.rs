use chatvcode_llm::ChatMessage;

use crate::types::TokenBudgetConfig;

pub trait TokenEstimator: Send + Sync {
    fn estimate_text(&self, text: &str) -> usize;

    fn estimate_messages(&self, messages: &[ChatMessage]) -> usize;
}

pub struct SimpleTokenEstimator;

impl SimpleTokenEstimator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleTokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenEstimator for SimpleTokenEstimator {
    fn estimate_text(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        (text.len() + 3) / 4
    }

    fn estimate_messages(&self, messages: &[ChatMessage]) -> usize {
        const OVERHEAD_PER_MSG: usize = 4;
        messages
            .iter()
            .map(|m| self.estimate_text(&m.content) + OVERHEAD_PER_MSG)
            .sum()
    }
}

pub struct SessionContext<'a> {
    pub system_prompt: &'a str,
    pub messages: &'a [ChatMessage],
}

pub struct TokenBudgetManager {
    config: TokenBudgetConfig,
    estimator: Box<dyn TokenEstimator>,
}

impl std::fmt::Debug for TokenBudgetManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenBudgetManager")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl TokenBudgetManager {
    pub fn new(config: TokenBudgetConfig) -> Self {
        Self { config, estimator: Box::new(SimpleTokenEstimator::new()) }
    }

    pub fn with_estimator(config: TokenBudgetConfig, estimator: Box<dyn TokenEstimator>) -> Self {
        Self { config, estimator }
    }

    pub fn config(&self) -> &TokenBudgetConfig {
        &self.config
    }

    pub fn used_tokens(&self, session: &SessionContext<'_>) -> usize {
        let system_tokens = self.estimator.estimate_text(session.system_prompt);
        let history_tokens = self.estimator.estimate_messages(session.messages);
        system_tokens + history_tokens
    }

    pub fn remaining_tokens(&self, session: &SessionContext<'_>) -> usize {
        let used = self.used_tokens(session);
        self.config
            .total_budget
            .saturating_sub(used)
            .saturating_sub(self.config.response_reserve)
    }

    pub fn can_add_message(&self, session: &SessionContext<'_>, message: &str) -> bool {
        let message_tokens = self.estimator.estimate_text(message);
        self.remaining_tokens(session) >= message_tokens
    }

    pub fn truncate_tool_result(&self, result: &str) -> String {
        let tokens = self.estimator.estimate_text(result);
        if tokens <= self.config.tool_result_max {
            result.to_string()
        } else {
            let target_chars = self.config.tool_result_max * 4;
            let truncated = &result[..target_chars.min(result.len())];
            format!(
                "{}...\n[Result truncated, {} tokens omitted]",
                truncated,
                tokens - self.config.tool_result_max
            )
        }
    }

    pub fn is_budget_low(&self, session: &SessionContext<'_>) -> bool {
        let remaining = self.remaining_tokens(session);
        remaining < self.config.response_reserve
    }

    pub fn budget_report(&self, session: &SessionContext<'_>) -> BudgetReport {
        let used = self.used_tokens(session);
        let remaining = self.remaining_tokens(session);
        let system_tokens = self.estimator.estimate_text(session.system_prompt);
        let history_tokens = self.estimator.estimate_messages(session.messages);
        BudgetReport {
            total: self.config.total_budget,
            used,
            remaining,
            history_tokens,
            system_tokens,
            is_low: remaining < self.config.response_reserve,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BudgetReport {
    pub total: usize,
    pub used: usize,
    pub remaining: usize,
    pub history_tokens: usize,
    pub system_tokens: usize,
    pub is_low: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_text_empty() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text(""), 0);
    }

    #[test]
    fn estimate_text_short() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text("hi"), 1);
    }

    #[test]
    fn estimate_text_exact_multiple_of_4() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text("hello!"), 2);
        assert_eq!(est.estimate_text("abcdefgh"), 2);
    }

    #[test]
    fn estimate_text_rounds_up() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_text("a"), 1);
        assert_eq!(est.estimate_text("abc"), 1);
        assert_eq!(est.estimate_text("abcde"), 2);
    }

    #[test]
    fn estimate_messages_empty() {
        let est = SimpleTokenEstimator::new();
        assert_eq!(est.estimate_messages(&[]), 0);
    }

    #[test]
    fn estimate_messages_single() {
        let est = SimpleTokenEstimator::new();
        let msgs = vec![ChatMessage::user("hello world!")];
        let tokens = est.estimate_messages(&msgs);
        assert_eq!(tokens, est.estimate_text("hello world!") + 4);
    }

    #[test]
    fn estimate_messages_multiple() {
        let est = SimpleTokenEstimator::new();
        let msgs = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello! How can I help?"),
        ];
        let tokens = est.estimate_messages(&msgs);
        let expected: usize = msgs.iter().map(|m| est.estimate_text(&m.content) + 4).sum();
        assert_eq!(tokens, expected);
    }

    #[test]
    fn simple_token_estimator_default() {
        let est = SimpleTokenEstimator::default();
        assert_eq!(est.estimate_text("test"), 1);
    }

    fn default_config() -> TokenBudgetConfig {
        TokenBudgetConfig {
            total_budget: 1000,
            system_prompt_reserve: 100,
            tool_result_max: 200,
            history_budget: 500,
            response_reserve: 200,
        }
    }

    fn empty_session<'a>() -> SessionContext<'a> {
        SessionContext { system_prompt: "", messages: &[] }
    }

    #[test]
    fn budget_manager_new() {
        let mgr = TokenBudgetManager::new(default_config());
        assert_eq!(mgr.config().total_budget, 1000);
    }

    #[test]
    fn used_tokens_empty_session() {
        let mgr = TokenBudgetManager::new(default_config());
        let session = empty_session();
        assert_eq!(mgr.used_tokens(&session), 0);
    }

    #[test]
    fn used_tokens_with_system_prompt() {
        let mgr = TokenBudgetManager::new(default_config());
        let prompt = "You are a helpful assistant.";
        let session = SessionContext { system_prompt: prompt, messages: &[] };
        let expected = SimpleTokenEstimator::new().estimate_text(prompt);
        assert_eq!(mgr.used_tokens(&session), expected);
    }

    #[test]
    fn used_tokens_with_messages() {
        let mgr = TokenBudgetManager::new(default_config());
        let msgs = vec![ChatMessage::user("hello"), ChatMessage::assistant("world")];
        let session = SessionContext { system_prompt: "sys", messages: &msgs };
        let est = SimpleTokenEstimator::new();
        let expected = est.estimate_text("sys") + est.estimate_messages(&msgs);
        assert_eq!(mgr.used_tokens(&session), expected);
    }

    #[test]
    fn remaining_tokens_empty_session() {
        let mgr = TokenBudgetManager::new(default_config());
        let session = empty_session();
        // total_budget(1000) - used(0) - response_reserve(200) = 800
        assert_eq!(mgr.remaining_tokens(&session), 800);
    }

    #[test]
    fn remaining_tokens_with_usage() {
        let mgr = TokenBudgetManager::new(default_config());
        let prompt = "a".repeat(400); // 100 tokens
        let session = SessionContext { system_prompt: &prompt, messages: &[] };
        // total(1000) - used(100) - reserve(200) = 700
        assert_eq!(mgr.remaining_tokens(&session), 700);
    }

    #[test]
    fn remaining_tokens_saturates_at_zero() {
        let config =
            TokenBudgetConfig { total_budget: 100, response_reserve: 200, ..default_config() };
        let mgr = TokenBudgetManager::new(config);
        let session = empty_session();
        assert_eq!(mgr.remaining_tokens(&session), 0);
    }

    #[test]
    fn can_add_message_fits() {
        let mgr = TokenBudgetManager::new(default_config());
        let session = empty_session();
        assert!(mgr.can_add_message(&session, "short"));
    }

    #[test]
    fn can_add_message_exceeds_budget() {
        let mgr = TokenBudgetManager::new(default_config());
        let session = empty_session();
        let huge = "a".repeat(10000); // 2500 tokens > remaining(800)
        assert!(!mgr.can_add_message(&session, &huge));
    }

    #[test]
    fn truncate_tool_result_within_limit() {
        let mgr = TokenBudgetManager::new(default_config());
        let short = "short result";
        let result = mgr.truncate_tool_result(short);
        assert_eq!(result, short);
    }

    #[test]
    fn truncate_tool_result_exceeds_limit() {
        let mgr = TokenBudgetManager::new(default_config());
        let long = "a".repeat(4000); // 1000 tokens > tool_result_max(200)
        let result = mgr.truncate_tool_result(&long);
        assert!(result.contains("[Result truncated"));
        assert!(result.contains("800 tokens omitted"));
        assert!(result.len() < long.len());
    }

    #[test]
    fn truncate_tool_result_exact_limit() {
        let mgr = TokenBudgetManager::new(default_config());
        let exact = "a".repeat(800); // 200 tokens == tool_result_max
        let result = mgr.truncate_tool_result(&exact);
        assert_eq!(result, exact);
    }

    #[test]
    fn is_budget_low_false_when_plenty() {
        let mgr = TokenBudgetManager::new(default_config());
        let session = empty_session();
        // remaining = 800, response_reserve = 200 => not low
        assert!(!mgr.is_budget_low(&session));
    }

    #[test]
    fn is_budget_low_true_when_near_limit() {
        let config =
            TokenBudgetConfig { total_budget: 500, response_reserve: 200, ..default_config() };
        let mgr = TokenBudgetManager::new(config);
        let prompt = "a".repeat(400); // 100 tokens
        let session = SessionContext { system_prompt: &prompt, messages: &[] };
        // remaining = 500 - 100 - 200 = 200, not < 200 => not low
        assert!(!mgr.is_budget_low(&session));

        let prompt2 = "a".repeat(404); // 101 tokens
        let session2 = SessionContext { system_prompt: &prompt2, messages: &[] };
        // remaining = 500 - 101 - 200 = 199, < 200 => low
        assert!(mgr.is_budget_low(&session2));
    }

    #[test]
    fn budget_report_empty_session() {
        let mgr = TokenBudgetManager::new(default_config());
        let session = empty_session();
        let report = mgr.budget_report(&session);
        assert_eq!(report.total, 1000);
        assert_eq!(report.used, 0);
        assert_eq!(report.remaining, 800);
        assert_eq!(report.system_tokens, 0);
        assert_eq!(report.history_tokens, 0);
        assert!(!report.is_low);
    }

    #[test]
    fn budget_report_with_usage() {
        let mgr = TokenBudgetManager::new(default_config());
        let prompt = "a".repeat(400);
        let msgs = vec![ChatMessage::user("hello")];
        let session = SessionContext { system_prompt: &prompt, messages: &msgs };
        let report = mgr.budget_report(&session);
        assert_eq!(report.total, 1000);
        assert!(report.used > 0);
        assert_eq!(report.system_tokens, 100);
        assert!(report.history_tokens > 0);
        assert_eq!(report.remaining, 1000 - report.used - 200);
        assert!(!report.is_low);
    }

    #[test]
    fn budget_report_is_low() {
        let config =
            TokenBudgetConfig { total_budget: 300, response_reserve: 200, ..default_config() };
        let mgr = TokenBudgetManager::new(config);
        let prompt = "a".repeat(400); // 100 tokens
        let session = SessionContext { system_prompt: &prompt, messages: &[] };
        // remaining = 300 - 100 - 200 = 0, < 200 => low
        let report = mgr.budget_report(&session);
        assert!(report.is_low);
        assert_eq!(report.remaining, 0);
    }

    #[test]
    fn with_estimator_custom() {
        struct DoubleEstimator;
        impl TokenEstimator for DoubleEstimator {
            fn estimate_text(&self, text: &str) -> usize {
                text.len() / 2
            }
            fn estimate_messages(&self, messages: &[ChatMessage]) -> usize {
                messages
                    .iter()
                    .map(|m| self.estimate_text(&m.content))
                    .sum()
            }
        }

        let config = default_config();
        let mgr = TokenBudgetManager::with_estimator(config, Box::new(DoubleEstimator));
        let session = SessionContext {
            system_prompt: "abcdefgh", // 4 tokens with DoubleEstimator
            messages: &[],
        };
        assert_eq!(mgr.used_tokens(&session), 4);
    }

    #[test]
    fn default_config_values() {
        let config = TokenBudgetConfig::default();
        assert_eq!(config.total_budget, 8192);
        assert_eq!(config.system_prompt_reserve, 1500);
        assert_eq!(config.tool_result_max, 2000);
        assert_eq!(config.history_budget, 4000);
        assert_eq!(config.response_reserve, 1500);
    }

    #[test]
    fn budget_report_consistency_with_individual_methods() {
        let mgr = TokenBudgetManager::new(default_config());
        let prompt = "You are a helpful coding assistant.";
        let msgs = vec![
            ChatMessage::user("What is Rust?"),
            ChatMessage::assistant("Rust is a systems programming language."),
        ];
        let session = SessionContext { system_prompt: prompt, messages: &msgs };

        let report = mgr.budget_report(&session);
        assert_eq!(report.used, mgr.used_tokens(&session));
        assert_eq!(report.remaining, mgr.remaining_tokens(&session));
        assert_eq!(report.total, mgr.config().total_budget);
        assert_eq!(report.is_low, mgr.is_budget_low(&session));
    }

    #[test]
    fn multi_turn_conversation_budget_tracking() {
        let mgr = TokenBudgetManager::new(default_config());

        let system = "You are a helpful assistant.";
        let turn1 = vec![ChatMessage::user("Hello")];
        let session1 = SessionContext { system_prompt: system, messages: &turn1 };
        let used1 = mgr.used_tokens(&session1);
        let remaining1 = mgr.remaining_tokens(&session1);
        assert!(remaining1 > 0);
        assert!(!mgr.is_budget_low(&session1));

        let turn2 = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there! How can I help you today?"),
            ChatMessage::user("Tell me about machine learning"),
            ChatMessage::assistant("Machine learning is a subset of AI..."),
        ];
        let session2 = SessionContext { system_prompt: system, messages: &turn2 };
        let used2 = mgr.used_tokens(&session2);
        let remaining2 = mgr.remaining_tokens(&session2);
        assert!(used2 > used1);
        assert!(remaining2 < remaining1);
    }

    #[test]
    fn can_add_message_exact_fit() {
        let config =
            TokenBudgetConfig { total_budget: 300, response_reserve: 200, ..default_config() };
        let mgr = TokenBudgetManager::new(config);
        let session = empty_session();
        // remaining = 300 - 0 - 200 = 100 tokens
        let exact_msg = "a".repeat(400); // 100 tokens
        assert!(mgr.can_add_message(&session, &exact_msg));

        let over_msg = "a".repeat(404); // 101 tokens
        assert!(!mgr.can_add_message(&session, &over_msg));
    }

    #[test]
    fn truncate_tool_result_empty_string() {
        let mgr = TokenBudgetManager::new(default_config());
        assert_eq!(mgr.truncate_tool_result(""), "");
    }

    #[test]
    fn truncate_tool_result_one_over_limit() {
        let mgr = TokenBudgetManager::new(default_config());
        // tool_result_max = 200 tokens = 800 chars
        let one_over = "a".repeat(804); // 201 tokens
        let result = mgr.truncate_tool_result(&one_over);
        assert!(result.contains("[Result truncated"));
        assert!(result.contains("1 tokens omitted"));
    }

    #[test]
    fn estimate_text_unicode() {
        let est = SimpleTokenEstimator::new();
        let chinese = "你好世界"; // 4 chars * 3 bytes = 12 bytes
        let tokens = est.estimate_text(chinese);
        assert_eq!(tokens, (chinese.len() + 3) / 4);
    }

    #[test]
    fn budget_fully_exhausted() {
        let config =
            TokenBudgetConfig { total_budget: 200, response_reserve: 200, ..default_config() };
        let mgr = TokenBudgetManager::new(config);
        let prompt = "a".repeat(400); // 100 tokens
        let session = SessionContext { system_prompt: &prompt, messages: &[] };
        // remaining = 200 - 100 - 200 = 0 (saturating)
        assert_eq!(mgr.remaining_tokens(&session), 0);
        assert!(mgr.is_budget_low(&session));
        assert!(!mgr.can_add_message(&session, "x"));

        let report = mgr.budget_report(&session);
        assert!(report.is_low);
        assert_eq!(report.remaining, 0);
    }

    #[test]
    fn used_tokens_system_plus_history() {
        let mgr = TokenBudgetManager::new(default_config());
        let system = "system prompt here";
        let msgs = vec![ChatMessage::user("question"), ChatMessage::assistant("answer")];
        let session = SessionContext { system_prompt: system, messages: &msgs };
        let est = SimpleTokenEstimator::new();
        let expected_system = est.estimate_text(system);
        let expected_history = est.estimate_messages(&msgs);
        let report = mgr.budget_report(&session);
        assert_eq!(report.system_tokens, expected_system);
        assert_eq!(report.history_tokens, expected_history);
        assert_eq!(report.used, expected_system + expected_history);
    }
}
