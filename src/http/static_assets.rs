//! Static serving for the built web console (`web/dist`): the hashed Vite
//! assets under `/assets/*` with an immutable cache header and an `index.html`
//! SPA fallback for client-side routes. The router is mounted only as the
//! unmatched-route fallback when `STATIC_DIR` names a directory at boot;
//! device and REST paths have their own routes and never reach it.

use std::convert::Infallible;
use std::path::{Path, PathBuf};

use axum::Router;
use axum::extract::{Request, State};
use axum::http::HeaderValue;
use axum::http::header::CACHE_CONTROL;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tower::ServiceExt;
use tower_http::services::{ServeDir, ServeFile};

use crate::error::AppError;

const ASSETS_CACHE_CONTROL: &str = "public, max-age=31536000, immutable";
const SPA_CACHE_CONTROL: &str = "no-cache";

pub(crate) fn routes(dir: &Path) -> Router<()> {
    Router::new()
        .route("/assets/{*path}", get(serve_asset))
        .fallback(serve_spa)
        .with_state(dir.to_path_buf())
}

async fn serve_asset(State(dir): State<PathBuf>, req: Request) -> Response {
    serve(ServeDir::new(&dir), req, ASSETS_CACHE_CONTROL).await
}

async fn serve_spa(State(dir): State<PathBuf>, req: Request) -> Response {
    let path = req.uri().path();
    if path == "/api"
        || path.starts_with("/api/")
        || path == "/gateway"
        || path.starts_with("/gateway/")
    {
        return AppError::NotFound.into_response();
    }
    let service = ServeDir::new(&dir).fallback(ServeFile::new(dir.join("index.html")));
    serve(service, req, SPA_CACHE_CONTROL).await
}

async fn serve<S>(service: S, req: Request, cache_control: &'static str) -> Response
where
    S: tower::Service<Request, Error = Infallible>,
    S::Response: IntoResponse,
{
    let mut response = match service.oneshot(req).await {
        Ok(response) => response.into_response(),
        Err(error) => match error {},
    };
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static(cache_control));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn fixture() -> TempDir {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("assets")).unwrap();
        std::fs::write(dir.path().join("index.html"), "<html>index</html>").unwrap();
        std::fs::write(dir.path().join("favicon.svg"), "<svg/>").unwrap();
        std::fs::write(dir.path().join("assets/app-1.js"), "console.log(1)").unwrap();
        dir
    }

    async fn get(app: Router<()>, uri: &str) -> Response {
        app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn index_serves_at_root_and_unknown_paths_with_no_cache() {
        let dir = fixture();
        let app = routes(dir.path());
        for uri in ["/", "/login", "/agents/1"] {
            let response = get(app.clone(), uri).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.headers()[CACHE_CONTROL],
                SPA_CACHE_CONTROL,
                "cache-control for {uri}"
            );
            assert!(
                response.headers()[axum::http::header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html")
            );
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert_eq!(body.as_ref(), b"<html>index</html>");
        }
    }

    #[tokio::test]
    async fn root_files_serve_with_no_cache() {
        let dir = fixture();
        let response = get(routes(dir.path()), "/favicon.svg").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CACHE_CONTROL], SPA_CACHE_CONTROL);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), b"<svg/>");
    }

    #[tokio::test]
    async fn hashed_assets_get_the_immutable_header() {
        let dir = fixture();
        let response = get(routes(dir.path()), "/assets/app-1.js").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CACHE_CONTROL], ASSETS_CACHE_CONTROL);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), b"console.log(1)");
    }

    #[tokio::test]
    async fn missing_assets_404_instead_of_the_spa_fallback() {
        let dir = fixture();
        let response = get(routes(dir.path()), "/assets/nope.js").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unmatched_api_and_gateway_paths_stay_json_404s() {
        let dir = fixture();
        let app = routes(dir.path());
        for uri in ["/api/nope", "/gateway/nope"] {
            let response = get(app.clone(), uri).await;
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "status for {uri}");
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert!(
                std::str::from_utf8(&body).unwrap().contains("\"error\""),
                "body for {uri}"
            );
        }
    }
}
