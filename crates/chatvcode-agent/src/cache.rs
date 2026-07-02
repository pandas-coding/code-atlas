use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chatvcode_llm::ToolResult;
use lru::LruCache;

#[derive(Debug)]
struct CacheEntry {
    result: ToolResult,
    inserted_at: Instant,
}

#[derive(Debug)]
pub struct ToolResultCache {
    cache: Mutex<LruCache<String, CacheEntry>>,
    ttl: Duration,
}

impl ToolResultCache {
    pub fn new(ttl: Duration, max_size: usize) -> Self {
        let cap = NonZeroUsize::new(max_size.max(1)).unwrap();
        Self { cache: Mutex::new(LruCache::new(cap)), ttl }
    }

    pub fn get(&self, key: &str) -> Option<ToolResult> {
        let mut cache = self.cache.lock().ok()?;
        let entry = cache.get(key)?;
        if entry.inserted_at.elapsed() > self.ttl {
            cache.pop(key);
            return None;
        }
        Some(entry.result.clone())
    }

    pub fn set(&self, key: String, result: ToolResult) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(key, CacheEntry { result, inserted_at: Instant::now() });
        }
    }

    pub fn len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

impl Default for ToolResultCache {
    fn default() -> Self {
        Self::new(Duration::from_secs(300), 128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn make_result(value: &str) -> ToolResult {
        ToolResult::success(Value::String(value.to_string()))
    }

    #[test]
    fn test_cache_hit() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 10);
        cache.set("read_file:path=a.rs".into(), make_result("content_a"));

        let hit = cache.get("read_file:path=a.rs");
        assert!(hit.is_some());
        let r = hit.unwrap();
        assert!(r.success);
        assert_eq!(r.value, Value::String("content_a".into()));
    }

    #[test]
    fn test_cache_miss() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 10);
        cache.set("read_file:path=a.rs".into(), make_result("content_a"));

        assert!(cache.get("read_file:path=b.rs").is_none());
    }

    #[test]
    fn test_cache_overwrite() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 10);
        cache.set("key".into(), make_result("v1"));
        cache.set("key".into(), make_result("v2"));

        let hit = cache.get("key").unwrap();
        assert_eq!(hit.value, Value::String("v2".into()));
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = ToolResultCache::new(Duration::from_millis(50), 10);
        cache.set("key".into(), make_result("value"));

        assert!(cache.get("key").is_some());
        std::thread::sleep(Duration::from_millis(80));
        assert!(cache.get("key").is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 2);
        cache.set("k1".into(), make_result("v1"));
        cache.set("k2".into(), make_result("v2"));
        cache.set("k3".into(), make_result("v3"));

        assert!(cache.get("k1").is_none());
        assert!(cache.get("k2").is_some());
        assert!(cache.get("k3").is_some());
    }

    #[test]
    fn test_lru_access_refreshes_order() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 2);
        cache.set("k1".into(), make_result("v1"));
        cache.set("k2".into(), make_result("v2"));

        let _ = cache.get("k1");

        cache.set("k3".into(), make_result("v3"));

        assert!(cache.get("k1").is_some());
        assert!(cache.get("k2").is_none());
        assert!(cache.get("k3").is_some());
    }

    #[test]
    fn test_clear() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 10);
        cache.set("a".into(), make_result("1"));
        cache.set("b".into(), make_result("2"));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_default() {
        let cache = ToolResultCache::default();
        assert!(cache.is_empty());
        cache.set("x".into(), make_result("y"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_error_results_cached() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 10);
        let err = ToolResult::error("file not found");
        cache.set("read_file:path=missing.rs".into(), err);

        let hit = cache.get("read_file:path=missing.rs").unwrap();
        assert!(!hit.success);
    }

    #[test]
    fn test_min_capacity_is_one() {
        let cache = ToolResultCache::new(Duration::from_secs(60), 0);
        cache.set("k".into(), make_result("v"));
        assert!(cache.get("k").is_some());
    }
}
