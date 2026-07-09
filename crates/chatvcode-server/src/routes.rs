//! HTTP 路由定义：集中声明各路径与方法的映射，便于扩展新端点。

/// 路由表中的单个条目。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Route {
    pub method: &'static str,
    pub path: &'static str,
    pub kind: RouteKind,
}

/// 路由类型：决定如何处理该请求。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteKind {
    /// 健康检查端点。
    Health,
    /// Agent 查询端点（SSE 流）。
    AgentQuery,
    /// 静态资源（Web UI），目前未实现。
    Static,
}

/// 内置路由表。
pub const ROUTES: &[Route] = &[
    Route { method: "GET", path: "/health", kind: RouteKind::Health },
    Route { method: "POST", path: "/agent/query", kind: RouteKind::AgentQuery },
];

/// 根据方法与路径查找路由。
pub fn resolve(method: &str, path: &str) -> Option<RouteKind> {
    ROUTES
        .iter()
        .find(|r| r.method == method && r.path == path)
        .map(|r| r.kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_health() {
        assert_eq!(resolve("GET", "/health"), Some(RouteKind::Health));
    }

    #[test]
    fn resolve_agent_query() {
        assert_eq!(resolve("POST", "/agent/query"), Some(RouteKind::AgentQuery));
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve("POST", "/health").is_none());
        assert!(resolve("GET", "/agent/query").is_none());
        assert!(resolve("DELETE", "/whatever").is_none());
    }

    #[test]
    fn route_table_includes_minimum_routes() {
        assert!(ROUTES.len() >= 2);
    }
}
