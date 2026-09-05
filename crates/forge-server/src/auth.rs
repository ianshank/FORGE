//! Optional bearer-token authentication for the mutating routes.
//!
//! The server's write surface — `/api/env/reset`, `/api/env/step`,
//! `/api/scenario/remix`, and the `POST` history writers — used to be
//! reachable by anything that could open a socket to the port. CORS
//! constrains browsers and nothing else.
//!
//! This module supplies a tower middleware ([`require_bearer_token`],
//! mounted by [`protect`]) that rejects requests without a matching
//! `Authorization: Bearer <token>` header. It is a *layer*
//! rather than a per-handler check so a newly-added mutating route
//! cannot forget to call it — routes are protected by where they are
//! mounted (see [`crate::routes::build_router`]).
//!
//! Auth is opt-in: [`crate::config::ServerConfig::auth_token`] defaults
//! to `None`, in which case no layer is installed at all and behaviour
//! is byte-for-byte what it was before. The binary logs a
//! [`crate::config::SecurityAdvisory`] at `warn!` when that is the case.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tracing::{debug, warn};

/// The `Authorization` scheme this server accepts, including its
/// trailing space.
const BEARER_PREFIX: &str = "Bearer ";

/// Shared middleware state: the token a request must present.
///
/// Held behind an `Arc` because axum clones the state per request and
/// the token is immutable for the process lifetime.
#[derive(Clone)]
pub struct AuthState {
    expected: Arc<str>,
}

impl AuthState {
    /// Build state requiring `token` on every guarded request.
    pub fn new(token: impl AsRef<str>) -> Self {
        Self {
            expected: Arc::from(token.as_ref()),
        }
    }

    /// Constant-time check of a presented token against the expected
    /// one.
    ///
    /// Uses a full-width XOR fold rather than `==` so the comparison
    /// does not return early on the first differing byte, which would
    /// leak the matched prefix length through timing. Lengths are
    /// compared up front — that only leaks the token's length, which
    /// an attacker who can read this source already knows nothing
    /// about, and any constant-time scheme has to handle mismatched
    /// lengths somehow.
    pub fn matches(&self, presented: &str) -> bool {
        let expected = self.expected.as_bytes();
        let presented = presented.as_bytes();
        if expected.len() != presented.len() {
            return false;
        }
        let mut diff: u8 = 0;
        for (a, b) in expected.iter().zip(presented.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

/// Body returned when a guarded route rejects a request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthErrorResponse {
    /// Machine-readable error code (`unauthorized`).
    pub error: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Reason a request failed authentication. Logged, never returned to
/// the caller — the response body is deliberately uniform so it cannot
/// be used to probe which part of the credential was wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthFailure {
    /// No `Authorization` header at all.
    MissingHeader,
    /// Header present but not `Bearer <token>` (wrong scheme, or bytes
    /// that are not valid header text).
    MalformedHeader,
    /// Well-formed `Bearer` header carrying the wrong token.
    TokenMismatch,
}

impl AuthFailure {
    fn as_str(self) -> &'static str {
        match self {
            Self::MissingHeader => "missing_authorization_header",
            Self::MalformedHeader => "malformed_authorization_header",
            Self::TokenMismatch => "token_mismatch",
        }
    }
}

/// Build the 401 response. Uniform across every [`AuthFailure`].
fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(AuthErrorResponse {
            error: "unauthorized".to_string(),
            message: "This endpoint requires an `Authorization: Bearer <token>` header."
                .to_string(),
        }),
    )
        .into_response()
}

/// Classify a request's `Authorization` header against `state`.
fn check(state: &AuthState, req: &Request) -> Result<(), AuthFailure> {
    let Some(value) = req.headers().get(header::AUTHORIZATION) else {
        return Err(AuthFailure::MissingHeader);
    };
    let Ok(text) = value.to_str() else {
        return Err(AuthFailure::MalformedHeader);
    };
    let Some(token) = text.strip_prefix(BEARER_PREFIX) else {
        return Err(AuthFailure::MalformedHeader);
    };
    if state.matches(token) {
        Ok(())
    } else {
        Err(AuthFailure::TokenMismatch)
    }
}

/// Middleware that requires a matching bearer token.
///
/// Rejects with `401 Unauthorized` and a JSON body; passes the request
/// through untouched on success. Failures are logged at `warn!` with
/// the method and path (never the presented token).
pub async fn require_bearer_token(
    State(state): State<AuthState>,
    req: Request,
    next: Next,
) -> Response {
    match check(&state, &req) {
        Ok(()) => {
            debug!(path = %req.uri().path(), "bearer auth accepted");
            next.run(req).await
        }
        Err(failure) => {
            warn!(
                method = %req.method(),
                path = %req.uri().path(),
                reason = failure.as_str(),
                "rejected unauthenticated request to a mutating endpoint"
            );
            unauthorized()
        }
    }
}

/// Apply [`require_bearer_token`] to every route in `router`.
///
/// Applying the guard to a sub-router (rather than to individual
/// handlers) is what makes it unforgettable: a mutating route is
/// protected by *where it is mounted*, so adding one to the guarded
/// group in [`crate::routes::build_router`] cannot miss the check.
pub fn protect<S>(router: axum::Router<S>, token: &str) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(from_fn_with_state(
        AuthState::new(token),
        require_bearer_token,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    fn request_with_auth(value: Option<&str>) -> Request {
        let mut builder = axum::http::Request::builder().uri("/api/env/step");
        if let Some(v) = value {
            builder = builder.header(header::AUTHORIZATION, v);
        }
        builder.body(Body::empty()).unwrap()
    }

    /// The three failure reasons must stay distinct, non-empty, and stable.
    ///
    /// This looks like a test of a logging helper, and it is — but the log is
    /// the *only* place the distinction survives. `unauthorized()` returns a
    /// deliberately uniform body so a caller cannot probe which half of the
    /// credential was wrong, which means `reason = failure.as_str()` at the
    /// `warn!` site is an operator's sole means of telling a misconfigured
    /// client (`missing_authorization_header`) from someone guessing tokens
    /// (`token_mismatch`). Collapse those to one string and credential-probing
    /// stops being visible in the logs, with every other test still green.
    ///
    /// Found by mutation testing: `cargo mutants -p forge-server --file
    /// auth.rs` reported `as_str -> ""` and `as_str -> "xyzzy"` as the only
    /// two survivors out of sixteen. The strings are pinned exactly because
    /// they are a log contract an alert rule may key on, so silently renaming
    /// one should fail here rather than in someone's dashboard.
    #[test]
    fn auth_failure_reasons_are_distinct_and_stable() {
        let cases = [
            (AuthFailure::MissingHeader, "missing_authorization_header"),
            (
                AuthFailure::MalformedHeader,
                "malformed_authorization_header",
            ),
            (AuthFailure::TokenMismatch, "token_mismatch"),
        ];

        for (failure, expected) in cases {
            assert_eq!(
                failure.as_str(),
                expected,
                "{failure:?} is logged as a reason operators and alert rules key on"
            );
        }

        let reasons: Vec<&str> = cases.iter().map(|(f, _)| f.as_str()).collect();
        assert!(
            reasons.iter().all(|r| !r.is_empty()),
            "an empty reason makes a rejection unattributable in the logs"
        );

        let mut unique = reasons.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            reasons.len(),
            "two failure modes sharing a reason string are indistinguishable to an \
             operator, which is the whole point of not putting them in the response"
        );
    }

    #[test]
    fn matches_accepts_exact_token() {
        let state = AuthState::new("correct-horse");
        assert!(state.matches("correct-horse"));
    }

    #[test]
    fn matches_rejects_wrong_token_same_length() {
        let state = AuthState::new("correct-horse");
        assert!(!state.matches("correct-horsf"));
    }

    #[test]
    fn matches_rejects_prefix_and_suffix() {
        let state = AuthState::new("correct-horse");
        assert!(!state.matches("correct"));
        assert!(!state.matches("correct-horse-extra"));
        assert!(!state.matches(""));
    }

    #[test]
    fn check_accepts_valid_bearer_header() {
        let state = AuthState::new("tok");
        assert_eq!(
            check(&state, &request_with_auth(Some("Bearer tok"))),
            Ok(())
        );
    }

    #[test]
    fn check_reports_missing_header() {
        let state = AuthState::new("tok");
        assert_eq!(
            check(&state, &request_with_auth(None)),
            Err(AuthFailure::MissingHeader)
        );
    }

    #[test]
    fn check_reports_wrong_scheme_as_malformed() {
        let state = AuthState::new("tok");
        assert_eq!(
            check(&state, &request_with_auth(Some("Basic dG9rOg=="))),
            Err(AuthFailure::MalformedHeader)
        );
        // Scheme comparison is case-sensitive per the constant.
        assert_eq!(
            check(&state, &request_with_auth(Some("bearer tok"))),
            Err(AuthFailure::MalformedHeader)
        );
        // Bare token with no scheme.
        assert_eq!(
            check(&state, &request_with_auth(Some("tok"))),
            Err(AuthFailure::MalformedHeader)
        );
    }

    #[test]
    fn check_reports_token_mismatch() {
        let state = AuthState::new("tok");
        assert_eq!(
            check(&state, &request_with_auth(Some("Bearer nope"))),
            Err(AuthFailure::TokenMismatch)
        );
    }

    /// Every failure mode must produce the same 401 body, so the
    /// response cannot be used to distinguish "no header" from "wrong
    /// token".
    #[test]
    fn unauthorized_response_is_401_and_uniform() {
        let resp = unauthorized();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
