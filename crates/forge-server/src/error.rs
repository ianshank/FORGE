//! Error types for the REST API surface.
//!
//! [`ApiError`] is a `thiserror` enum that maps cleanly onto HTTP status codes
//! via its [`IntoResponse`] implementation, replacing ad-hoc per-handler error
//! handling.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

/// Errors returned by the REST API handlers.
#[derive(Debug, Error)]
pub enum ApiError {
    /// The supplied configuration was invalid (e.g. world construction failed).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The session environment has not been reset yet.
    #[error("environment not reset; call /api/env/reset first")]
    NotReset,

    /// The requested action id is outside the valid action space.
    #[error("invalid action id: {0}")]
    InvalidAction(u32),

    /// An unexpected internal error (e.g. a poisoned lock).
    #[error("internal error: {0}")]
    Internal(String),
}

/// JSON body returned for an error response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorBody {
    /// Human-readable error message.
    error: String,
}

impl ApiError {
    /// Returns the HTTP status code this error maps to.
    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::Config(_) => StatusCode::BAD_REQUEST,
            ApiError::NotReset => StatusCode::CONFLICT,
            ApiError::InvalidAction(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = ErrorBody {
            error: self.to_string(),
        };
        if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %self, "REST API internal error");
        } else {
            tracing::debug!(status = %status, error = %self, "REST API error");
        }
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_mapping() {
        assert_eq!(
            ApiError::Config("x".into()).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(ApiError::NotReset.status(), StatusCode::CONFLICT);
        assert_eq!(
            ApiError::InvalidAction(99).status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(
            ApiError::Internal("boom".into()).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_display_messages() {
        assert_eq!(
            ApiError::InvalidAction(7).to_string(),
            "invalid action id: 7"
        );
        assert!(ApiError::NotReset
            .to_string()
            .contains("call /api/env/reset"));
    }

    #[test]
    fn test_into_response_status() {
        let resp = ApiError::NotReset.into_response();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }
}
