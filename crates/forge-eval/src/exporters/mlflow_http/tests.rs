use super::*;

#[test]
fn retry_policy_table_pins_retryable_statuses() {
    for (status, want) in [
        (200u16, false),
        (201, false),
        (204, false),
        (301, false),
        (400, false),
        (401, false),
        (403, false),
        (404, false),
        (408, true),
        (409, false),
        (422, false),
        (429, true),
        (500, true),
        (502, true),
        (503, true),
        (504, true),
        (599, true),
    ] {
        assert_eq!(RetryPolicy::should_retry(status), want, "status={status}");
    }
}

#[test]
fn retry_policy_wait_ms_is_exponential_with_cap() {
    let p = RetryPolicy {
        max_retries: 10,
        backoff_base_ms: 250,
    };
    // 250 * 2^1 = 500; 250 * 2^2 = 1000; 250 * 2^3 = 2000; ...
    assert_eq!(p.wait_ms(1), 500);
    assert_eq!(p.wait_ms(2), 1_000);
    assert_eq!(p.wait_ms(3), 2_000);
    assert_eq!(p.wait_ms(4), 4_000);
    // Cap at 60s — wait_ms(20) would overflow without the cap.
    assert_eq!(p.wait_ms(20), DEFAULT_HTTP_BACKOFF_CAP_MS);
    assert_eq!(p.wait_ms(50), DEFAULT_HTTP_BACKOFF_CAP_MS);
}

#[test]
fn http_client_config_default_uses_constants() {
    let cfg = HttpClientConfig::default();
    assert_eq!(cfg.timeout_ms, DEFAULT_HTTP_TIMEOUT_MS);
    assert_eq!(cfg.max_retries, DEFAULT_HTTP_MAX_RETRIES);
    assert_eq!(cfg.backoff_base_ms, DEFAULT_HTTP_BACKOFF_BASE_MS);
    assert_eq!(cfg.batch_size, DEFAULT_LOG_BATCH_SIZE);
    assert_eq!(cfg.user_agent, DEFAULT_USER_AGENT);
    assert!(cfg.bearer_token.is_none());
    assert!(
        cfg.batch_size <= 1_000,
        "MLflow REST caps log_batch at 1000"
    );
}

#[test]
fn client_new_accepts_valid_url_and_stores_config() {
    let base = Url::parse("http://localhost:5000").unwrap();
    let client = MlflowHttpClient::new(base.clone(), HttpClientConfig::default()).unwrap();
    assert_eq!(client.base().as_str(), base.as_str());
    assert_eq!(client.batch_size(), DEFAULT_LOG_BATCH_SIZE);
    assert!(!client.has_auth());
}

#[test]
fn client_debug_redacts_bearer_token() {
    let base = Url::parse("https://mlflow.example").unwrap();
    let cfg = HttpClientConfig {
        bearer_token: Some("super-secret-PAT".to_string()),
        ..HttpClientConfig::default()
    };
    let client = MlflowHttpClient::new(base, cfg).unwrap();
    let debug = format!("{client:?}");
    assert!(
        !debug.contains("super-secret-PAT"),
        "bearer token must not leak into Debug output: {debug}"
    );
    assert!(debug.contains("auth_present: true"));
}

#[test]
fn classify_status_distinguishes_retryable_from_terminal() {
    assert!(classify_status(StatusCode::OK, "").is_ok());
    assert!(classify_status(StatusCode::CREATED, "").is_ok());

    let retryable = classify_status(StatusCode::SERVICE_UNAVAILABLE, "down");
    assert!(matches!(retryable, Err(ExportError::Retryable(_))));

    let terminal = classify_status(StatusCode::BAD_REQUEST, "bad");
    assert!(matches!(terminal, Err(ExportError::Http(_))));
}

// ─── REST method tests (mockito) ───────────────────────────────────────
//
// Each test uses a fast retry config (1ms base, 0 retries) so even a
// failing test exits in milliseconds rather than sitting on the
// default 250ms × 2^n backoff schedule.

use mockito::{Matcher, Server};
use serde_json::json;

fn fast_client(server: &Server) -> MlflowHttpClient {
    let cfg = HttpClientConfig {
        timeout_ms: 5_000,
        max_retries: 0,
        backoff_base_ms: 1,
        batch_size: DEFAULT_LOG_BATCH_SIZE,
        bearer_token: None,
        user_agent: DEFAULT_USER_AGENT.to_string(),
    };
    MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap()
}

#[test]
fn get_or_create_experiment_returns_existing_id_on_hit() {
    let mut server = Server::new();
    let mock = server
        .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
        .match_query(Matcher::UrlEncoded(
            "experiment_name".into(),
            "exp-1".into(),
        ))
        .with_status(200)
        .with_body(json!({"experiment": {"experiment_id": "42"}}).to_string())
        .create();
    let client = fast_client(&server);
    let id = client.get_or_create_experiment("exp-1").unwrap();
    assert_eq!(id, "42");
    mock.assert();
}

#[test]
fn get_or_create_experiment_falls_back_to_create_on_404() {
    let mut server = Server::new();
    let lookup = server
        .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
        // mockito requires an explicit query matcher; default mocks don't
        // match requests that carry a query string.
        .match_query(Matcher::UrlEncoded(
            "experiment_name".into(),
            "missing-exp".into(),
        ))
        .with_status(404)
        .with_body(json!({"error_code": "RESOURCE_DOES_NOT_EXIST"}).to_string())
        .create();
    let create = server
        .mock("POST", "/api/2.0/mlflow/experiments/create")
        .match_body(Matcher::PartialJson(json!({"name": "missing-exp"})))
        .with_status(200)
        .with_body(json!({"experiment_id": "99"}).to_string())
        .create();
    let client = fast_client(&server);
    let id = client.get_or_create_experiment("missing-exp").unwrap();
    assert_eq!(id, "99");
    lookup.assert();
    create.assert();
}

#[test]
fn create_run_posts_experiment_id_and_parses_run_id() {
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/api/2.0/mlflow/runs/create")
        .with_status(200)
        .with_body(json!({"run": {"info": {"run_id": "abc123"}}}).to_string())
        .create();
    let client = fast_client(&server);
    let tags = vec![TagKv {
        key: "mlflow.runName".to_string(),
        value: "test".to_string(),
    }];
    let run_id = client.create_run("0", 1_000, &tags).unwrap();
    assert_eq!(run_id, "abc123");
    mock.assert();
}

#[test]
fn log_batch_chunks_metrics_at_batch_size_and_sends_params_only_once() {
    let mut server = Server::new();
    // 5 metrics, batch_size=2 → 3 chunks. Params + tags should only appear
    // in the first chunk's body.
    let mock = server
        .mock("POST", "/api/2.0/mlflow/runs/log-batch")
        .with_status(200)
        .expect(3)
        .with_body("{}")
        .create();
    let cfg = HttpClientConfig {
        batch_size: 2,
        max_retries: 0,
        backoff_base_ms: 1,
        ..HttpClientConfig::default()
    };
    let client = MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap();
    let metrics: Vec<MetricSample> = (0..5)
        .map(|i| MetricSample {
            key: format!("m{i}"),
            value: i as f64,
            timestamp_ms: 1,
            step: i,
        })
        .collect();
    let params = vec![ParamKv {
        key: "p".to_string(),
        value: "v".to_string(),
    }];
    let tags = vec![TagKv {
        key: "t".to_string(),
        value: "v".to_string(),
    }];
    client.log_batch("run-1", &metrics, &params, &tags).unwrap();
    mock.assert();
}

#[test]
fn log_batch_no_op_on_empty_input() {
    let server = Server::new(); // No mock — must not be hit.
    let client = fast_client(&server);
    client.log_batch("run-1", &[], &[], &[]).unwrap();
}

#[test]
fn set_tag_posts_run_id_and_kv() {
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/api/2.0/mlflow/runs/set-tag")
        .with_status(200)
        .with_body("{}")
        .match_body(Matcher::PartialJson(
            json!({"run_id": "r", "key": "k", "value": "v"}),
        ))
        .create();
    let client = fast_client(&server);
    client.set_tag("r", "k", "v").unwrap();
    mock.assert();
}

#[test]
fn set_terminated_emits_status_string_for_finished_and_failed() {
    for (status, wire) in [
        (RunStatus::Finished, "FINISHED"),
        (RunStatus::Failed, "FAILED"),
    ] {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/api/2.0/mlflow/runs/update")
            .with_status(200)
            .with_body("{}")
            .match_body(Matcher::PartialJson(
                json!({"run_id": "r", "status": wire, "end_time": 999u64}),
            ))
            .create();
        let client = fast_client(&server);
        client.set_terminated("r", status, 999).unwrap();
        mock.assert();
    }
}

#[test]
fn log_artifact_puts_bytes_under_rel_path_with_run_id_query() {
    let mut server = Server::new();
    let mock = server
        .mock("PUT", "/api/2.0/mlflow-artifacts/artifacts/scorecard.json")
        .match_query(Matcher::UrlEncoded("run_id".into(), "abc".into()))
        .with_status(200)
        .with_body("{}")
        .match_body(Matcher::Exact(r#"{"score":0.5}"#.to_string()))
        .create();
    let client = fast_client(&server);
    client
        .log_artifact("abc", "scorecard.json", br#"{"score":0.5}"#)
        .unwrap();
    mock.assert();
}

#[test]
fn log_artifact_handles_nested_rel_path_segments() {
    let mut server = Server::new();
    let mock = server
        .mock(
            "PUT",
            "/api/2.0/mlflow-artifacts/artifacts/replays/ep0/data.bin",
        )
        .match_query(Matcher::UrlEncoded("run_id".into(), "run-x".into()))
        .with_status(200)
        .with_body("{}")
        .create();
    let client = fast_client(&server);
    client
        .log_artifact("run-x", "replays/ep0/data.bin", b"binary")
        .unwrap();
    mock.assert();
}

#[test]
fn execute_with_retry_surfaces_retryable_after_budget_exhausted() {
    // Server returns 503 every call. With max_retries=2 we expect 3
    // attempts (1 initial + 2 retries) before giving up with Retryable.
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/api/2.0/mlflow/runs/set-tag")
        .with_status(503)
        .with_body("server down")
        .expect(3)
        .create();
    let cfg = HttpClientConfig {
        max_retries: 2,
        backoff_base_ms: 1,
        ..HttpClientConfig::default()
    };
    let client = MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap();
    let err = client.set_tag("r", "k", "v").unwrap_err();
    assert!(matches!(err, ExportError::Retryable(_)), "got {err:?}");
    mock.assert();
}

#[test]
fn execute_with_retry_surfaces_http_immediately_on_terminal_4xx() {
    // 400 is non-retryable. Even with max_retries=5 we should see
    // exactly one attempt, ending in ExportError::Http.
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/api/2.0/mlflow/runs/create")
        .with_status(400)
        .with_body("malformed")
        .expect(1)
        .create();
    let cfg = HttpClientConfig {
        max_retries: 5,
        backoff_base_ms: 1,
        ..HttpClientConfig::default()
    };
    let client = MlflowHttpClient::new(Url::parse(&server.url()).unwrap(), cfg).unwrap();
    let err = client.create_run("0", 0, &[]).unwrap_err();
    assert!(matches!(err, ExportError::Http(_)), "got {err:?}");
    mock.assert();
}

#[test]
fn run_status_wire_strings_pin_mlflow_enum() {
    // MLflow's REST contract for runs/update.status is a tiny enum;
    // pin the wire strings so a typo never silently breaks the UI.
    assert_eq!(RunStatus::Running.as_str(), "RUNNING");
    assert_eq!(RunStatus::Finished.as_str(), "FINISHED");
    assert_eq!(RunStatus::Failed.as_str(), "FAILED");
    assert_eq!(RunStatus::Killed.as_str(), "KILLED");
    assert_eq!(RunStatus::Scheduled.as_str(), "SCHEDULED");
}

// ─── MlflowHttpSink tests ──────────────────────────────────────────────

use crate::manifest::{RunManifest, MANIFEST_SOURCE_NAME, UNKNOWN};
use crate::scorecard::{ScenarioResult, Scorecard, SummaryStats, TierScore};
use chrono::Utc;
use forge_types::agent_interface::AgentMetadata;

fn fixture_scorecard_with_n_scenarios(n: usize) -> Scorecard {
    let scenarios: Vec<ScenarioResult> = (0..n)
        .map(|i| ScenarioResult::from_episodes(format!("s{i}"), 1, vec![]))
        .collect();
    Scorecard {
        agent_metadata: AgentMetadata::heuristic("TestAgent"),
        timestamp: "1970-01-01T00:00:00Z".to_string(),
        overall_score: 0.0,
        tier_scores: vec![TierScore {
            tier: 1,
            success_rate: 0.0,
            mean_reward: 0.0,
            mean_steps_to_completion: 0.0,
            episodes_evaluated: 0,
            scenarios_count: n as u32,
        }],
        scenario_results: scenarios,
        summary: SummaryStats {
            total_episodes: 0,
            total_steps: 0,
            wall_clock_seconds: 0.0,
            mean_decision_latency_ms: 0.0,
        },
    }
}

fn fixture_manifest() -> RunManifest {
    RunManifest {
        run_id: "parent-run".to_string(),
        experiment_name: "test-exp".to_string(),
        timestamp: Utc::now(),
        git_sha: UNKNOWN.to_string(),
        git_branch: UNKNOWN.to_string(),
        rustc_version: UNKNOWN.to_string(),
        user: UNKNOWN.to_string(),
        config_hash: "h".to_string(),
        scenario_file_hashes: vec![],
        source_name: MANIFEST_SOURCE_NAME.to_string(),
    }
}

fn http_sink_from_url(url: &str, max_retries: u32) -> MlflowHttpSink {
    // Struct-literal init keeps clippy's field_reassign_with_default
    // happy while still inheriting Default::default() for every field
    // the tests don't care about.
    let cfg = EvalConfig {
        mlflow_http_tracking_uri: Some(url.to_string()),
        mlflow_http_max_retries: max_retries,
        mlflow_http_backoff_base_ms: 1,
        experiment_name: Some("test-exp".to_string()),
        ..EvalConfig::default()
    };
    MlflowHttpSink::from_config(&cfg).unwrap()
}

/// Happy-path: 0 scenarios = 1 parent run only. Asserts the full
/// create-run → log-batch → log-artifact (4 inline) → set-terminated
/// FINISHED sequence happens exactly once and the experiment is
/// resolved up front.
#[test]
fn mlflow_http_sink_writes_parent_run_only_when_no_scenarios() {
    let mut server = Server::new();
    let _get_exp = server
        .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
        .match_query(Matcher::UrlEncoded(
            "experiment_name".into(),
            "test-exp".into(),
        ))
        .with_status(200)
        .with_body(json!({"experiment": {"experiment_id": "0"}}).to_string())
        .create();
    let _create_run = server
        .mock("POST", "/api/2.0/mlflow/runs/create")
        .with_status(200)
        .with_body(json!({"run": {"info": {"run_id": "server-parent"}}}).to_string())
        .expect(1)
        .create();
    let _log_batch = server
        .mock("POST", "/api/2.0/mlflow/runs/log-batch")
        .with_status(200)
        .with_body("{}")
        .expect(1)
        .create();
    // 4 inline artefacts on the parent: scorecard.json, scorecard.md,
    // manifest.json, tier_success_rates.html.
    let _log_artifact = server
        .mock(
            "PUT",
            Matcher::Regex(r"^/api/2\.0/mlflow-artifacts/artifacts/.+$".to_string()),
        )
        .match_query(Matcher::UrlEncoded("run_id".into(), "server-parent".into()))
        .with_status(200)
        .expect(4)
        .create();
    let set_terminated_finished = server
        .mock("POST", "/api/2.0/mlflow/runs/update")
        .match_body(Matcher::PartialJson(json!({
            "run_id": "server-parent",
            "status": "FINISHED",
        })))
        .with_status(200)
        .with_body("{}")
        .expect(1)
        .create();
    let sink = http_sink_from_url(&server.url(), 0);
    sink.export(
        &fixture_scorecard_with_n_scenarios(0),
        &fixture_manifest(),
        Path::new(""),
    )
    .unwrap();
    set_terminated_finished.assert();
}

/// Direct happy-path test for `create_experiment` (without the 404
/// fallback dance). Ensures the body field name + response-parse path
/// don't depend on `get_or_create_experiment` to remain wired.
#[test]
fn create_experiment_posts_name_and_parses_id() {
    let mut server = Server::new();
    let mock = server
        .mock("POST", "/api/2.0/mlflow/experiments/create")
        .match_body(Matcher::PartialJson(json!({"name": "fresh-exp"})))
        .with_status(200)
        .with_body(json!({"experiment_id": "777"}).to_string())
        .create();
    let client = fast_client(&server);
    let id = client.create_experiment("fresh-exp").unwrap();
    assert_eq!(id, "777");
    mock.assert();
}

/// `from_config` must reject a missing tracking URI with InvalidTarget
/// rather than panicking or returning a misleading transport error.
#[test]
fn mlflow_http_sink_from_config_errors_on_missing_uri() {
    let cfg = EvalConfig {
        mlflow_http_tracking_uri: None,
        ..EvalConfig::default()
    };
    let err = MlflowHttpSink::from_config(&cfg).unwrap_err();
    assert!(matches!(err, ExportError::InvalidTarget(_)), "got {err:?}");
}

/// `from_config` must reject a malformed URI with InvalidTarget.
#[test]
fn mlflow_http_sink_from_config_errors_on_malformed_uri() {
    let cfg = EvalConfig {
        mlflow_http_tracking_uri: Some("not a url".to_string()),
        ..EvalConfig::default()
    };
    let err = MlflowHttpSink::from_config(&cfg).unwrap_err();
    assert!(matches!(err, ExportError::InvalidTarget(_)), "got {err:?}");
}

/// `from_config` must reject non-HTTP(S) schemes. `Url::parse` will
/// happily accept things like `file:///tmp/x` and `ftp://h/p`, but
/// the MLflow REST contract only speaks HTTP — the sink would
/// otherwise fail much later with an opaque transport error.
#[test]
fn mlflow_http_sink_from_config_rejects_non_http_schemes() {
    for bad in [
        "file:///tmp/mlflow",
        "ftp://mlflow.example.com",
        "ws://mlflow.example.com",
    ] {
        let cfg = EvalConfig {
            mlflow_http_tracking_uri: Some(bad.to_string()),
            ..EvalConfig::default()
        };
        let err = MlflowHttpSink::from_config(&cfg).unwrap_err();
        assert!(
            matches!(err, ExportError::InvalidTarget(_)),
            "scheme {bad} must be InvalidTarget, got {err:?}",
        );
    }
}

/// `from_config` with no `experiment_name` falls back to the
/// canonical [`DEFAULT_EXPERIMENT_NAME`] constant.
#[test]
fn mlflow_http_sink_from_config_defaults_experiment_name() {
    let mut server = Server::new();
    let cfg = EvalConfig {
        mlflow_http_tracking_uri: Some(server.url()),
        mlflow_http_max_retries: 0,
        mlflow_http_backoff_base_ms: 1,
        experiment_name: None,
        ..EvalConfig::default()
    };
    let sink = MlflowHttpSink::from_config(&cfg).unwrap();
    assert_eq!(sink.experiment_name(), DEFAULT_EXPERIMENT_NAME);
    // Make sure we hit a mock so the test doesn't accidentally pass on
    // server.url() being unreachable.
    let _ = server
        .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
        .match_query(Matcher::UrlEncoded(
            "experiment_name".into(),
            DEFAULT_EXPERIMENT_NAME.into(),
        ))
        .with_status(200)
        .with_body(json!({"experiment": {"experiment_id": "0"}}).to_string())
        .create();
    sink.client
        .get_or_create_experiment(sink.experiment_name())
        .unwrap();
}

/// `upload_dir_recursively` exercise: nested tree → one log_artifact
/// PUT per file with paths anchored under `rel_root`.
#[test]
fn upload_dir_recursively_walks_nested_files() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("a.txt"), b"A").unwrap();
    std::fs::write(tmp.path().join("sub/b.txt"), b"B").unwrap();

    let mut server = Server::new();
    let mock_a = server
        .mock("PUT", "/api/2.0/mlflow-artifacts/artifacts/replays/a.txt")
        .match_query(Matcher::UrlEncoded("run_id".into(), "r1".into()))
        .with_status(200)
        .expect(1)
        .create();
    let mock_b = server
        .mock(
            "PUT",
            "/api/2.0/mlflow-artifacts/artifacts/replays/sub/b.txt",
        )
        .match_query(Matcher::UrlEncoded("run_id".into(), "r1".into()))
        .with_status(200)
        .expect(1)
        .create();
    let client = fast_client(&server);
    upload_dir_recursively(&client, "r1", "replays", tmp.path()).unwrap();
    mock_a.assert();
    mock_b.assert();
}

/// `upload_dir_recursively` must no-op on a missing dir (parity with
/// FsSink's `copy_subdir_if_exists`).
#[test]
fn upload_dir_recursively_no_op_on_missing_dir() {
    let server = Server::new(); // No mocks — must not be hit.
    let client = fast_client(&server);
    upload_dir_recursively(&client, "r1", "absent", Path::new("/no/such/dir")).unwrap();
}

/// Pin every const-block value to its on-disk-contract string so a
/// rename surfaces here, not at runtime when MLflow rejects the request.
/// Mirrors `exporters::tests::exporter_string_constants_are_stable_contract`
/// for the HTTP transport's own constants.
#[test]
fn mlflow_http_constants_are_stable_contract() {
    assert_eq!(API_ROOT, "api");
    assert_eq!(API_VERSION, "2.0");
    assert_eq!(NAMESPACE_MLFLOW, "mlflow");
    assert_eq!(NAMESPACE_ARTIFACTS, "mlflow-artifacts");
    assert_eq!(ARTIFACTS_PATH, "artifacts");
    assert_eq!(DEFAULT_EXPERIMENT_NAME, "forge-eval-default");
    // REST body / query fields.
    assert_eq!(FIELD_EXPERIMENT_NAME, "experiment_name");
    assert_eq!(FIELD_EXPERIMENT_ID, "experiment_id");
    assert_eq!(FIELD_RUN_ID, "run_id");
    assert_eq!(FIELD_RUN, "run");
    assert_eq!(FIELD_INFO, "info");
    assert_eq!(FIELD_START_TIME, "start_time");
    assert_eq!(FIELD_END_TIME, "end_time");
    assert_eq!(FIELD_STATUS, "status");
    assert_eq!(FIELD_KEY, "key");
    assert_eq!(FIELD_VALUE, "value");
    assert_eq!(FIELD_NAME, "name");
    assert_eq!(FIELD_TAGS, "tags");
    assert_eq!(FIELD_METRICS, "metrics");
    assert_eq!(FIELD_PARAMS, "params");
    assert_eq!(FIELD_TIMESTAMP, "timestamp");
    assert_eq!(FIELD_STEP, "step");
    assert_eq!(QUERY_RUN_ID, "run_id");
    // Response paths.
    assert_eq!(RESP_PATH_EXPERIMENT_ID, &["experiment", "experiment_id"]);
    assert_eq!(RESP_PATH_CREATE_EXP_ID, &["experiment_id"]);
    assert_eq!(RESP_PATH_RUN_ID, &["run", "info", "run_id"]);
}

/// Pin the binding between `config.rs`'s `DEFAULT_MLFLOW_HTTP_*` re-exports
/// (when feature is enabled) and `mlflow_http.rs`'s canonical constants.
/// Catches future drift between the two.
#[test]
fn config_re_exports_match_canonical_http_defaults() {
    assert_eq!(
        crate::config::DEFAULT_MLFLOW_HTTP_TIMEOUT_MS,
        DEFAULT_HTTP_TIMEOUT_MS
    );
    assert_eq!(
        crate::config::DEFAULT_MLFLOW_HTTP_MAX_RETRIES,
        DEFAULT_HTTP_MAX_RETRIES
    );
    assert_eq!(
        crate::config::DEFAULT_MLFLOW_HTTP_BACKOFF_BASE_MS,
        DEFAULT_HTTP_BACKOFF_BASE_MS
    );
    assert_eq!(
        crate::config::DEFAULT_MLFLOW_HTTP_BATCH_SIZE,
        DEFAULT_LOG_BATCH_SIZE
    );
}

/// Scopeguard contract: when a mid-export call (log_batch here) returns
/// the retry budget is exhausted, the sink propagates Err — AND the
/// run lands in FAILED, not RUNNING. Without the scopeguard the run
/// would be wedged in RUNNING in the UI.
#[test]
fn mlflow_http_sink_scopeguard_marks_run_failed_on_mid_export_error() {
    let mut server = Server::new();
    let _get_exp = server
        .mock("GET", "/api/2.0/mlflow/experiments/get-by-name")
        .match_query(Matcher::UrlEncoded(
            "experiment_name".into(),
            "test-exp".into(),
        ))
        .with_status(200)
        .with_body(json!({"experiment": {"experiment_id": "0"}}).to_string())
        .create();
    let _create_run = server
        .mock("POST", "/api/2.0/mlflow/runs/create")
        .with_status(200)
        .with_body(json!({"run": {"info": {"run_id": "doomed-run"}}}).to_string())
        .expect(1)
        .create();
    // Every log-batch attempt returns 503 → after the retry budget is
    // exhausted, the sink's `?` propagates and the scopeguard fires.
    let _log_batch_fail = server
        .mock("POST", "/api/2.0/mlflow/runs/log-batch")
        .with_status(503)
        .with_body("server down")
        .create();
    // The set_terminated(FAILED, ...) call the scopeguard MUST issue.
    let set_terminated_failed = server
        .mock("POST", "/api/2.0/mlflow/runs/update")
        .match_body(Matcher::PartialJson(json!({
            "run_id": "doomed-run",
            "status": "FAILED",
        })))
        .with_status(200)
        .with_body("{}")
        .expect(1)
        .create();
    let sink = http_sink_from_url(&server.url(), 0);
    let result = sink.export(
        &fixture_scorecard_with_n_scenarios(0),
        &fixture_manifest(),
        Path::new(""),
    );
    assert!(
        matches!(result, Err(ExportError::Retryable(_))),
        "got {result:?}"
    );
    // The critical contract: FAILED was sent — run isn't wedged in RUNNING.
    set_terminated_failed.assert();
}

/// Pin the child-run linkage contract: when given a parent's
/// SERVER-assigned id, the helper rewrites `mlflow.parentRunId` in
/// every child's tag set to that id and leaves all other tags
/// untouched. Pure-function unit test — no mockito, no server, no
/// parallelism contention. The sink's child-create flow consumes
/// this helper directly, so the linkage contract is enforced
/// transitively in `write_run`.
#[test]
fn rewrite_parent_run_id_tag_replaces_only_target_tag_for_children() {
    let original = vec![
        TagKv {
            key: "mlflow.parentRunId".to_string(),
            value: "PAYLOAD-PARENT".to_string(),
        },
        TagKv {
            key: "mlflow.runName".to_string(),
            value: "scenario_a".to_string(),
        },
        TagKv {
            key: "forge.eval.tier".to_string(),
            value: "1".to_string(),
        },
    ];

    // Child case: target tag is rewritten, others pass through.
    let rewritten = rewrite_parent_run_id_tag(&original, Some("SERVER-PARENT"));
    assert_eq!(rewritten.len(), 3);
    assert_eq!(rewritten[0].key, "mlflow.parentRunId");
    assert_eq!(
        rewritten[0].value, "SERVER-PARENT",
        "child must carry the parent's SERVER-assigned id, not the payload id"
    );
    assert_eq!(rewritten[1].value, "scenario_a", "other tags untouched");
    assert_eq!(rewritten[2].value, "1", "other tags untouched");

    // Parent case: passthrough (no rewrite).
    let unchanged = rewrite_parent_run_id_tag(&original, None);
    assert_eq!(unchanged.len(), 3);
    assert_eq!(unchanged[0].value, "PAYLOAD-PARENT");

    // Multiple parentRunId tags (shouldn't happen but mustn't panic).
    let pathological = vec![
        TagKv {
            key: "mlflow.parentRunId".to_string(),
            value: "A".to_string(),
        },
        TagKv {
            key: "mlflow.parentRunId".to_string(),
            value: "B".to_string(),
        },
    ];
    let both = rewrite_parent_run_id_tag(&pathological, Some("REWRITE"));
    assert!(both.iter().all(|t| t.value == "REWRITE"));
}
