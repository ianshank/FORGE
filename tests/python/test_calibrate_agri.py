"""Tests for scripts/calibrate_agri.py.

Runs all calibration functions in dry-run mode (no network downloads)
and verifies correctness of the fixed-point conversions, config merging,
and JSON/TOML export logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import calibrate_agri as cal


# ---------------------------------------------------------------------------
# Constants and fixed-point arithmetic
# ---------------------------------------------------------------------------


def test_fixed_scale_value() -> None:
    assert cal.FIXED_SCALE == 65536


def test_to_fixed_zero() -> None:
    assert cal.to_fixed(0.0) == 0


def test_to_fixed_one() -> None:
    assert cal.to_fixed(1.0) == 65536


def test_to_fixed_half() -> None:
    assert cal.to_fixed(0.5) == 32768


def test_to_fixed_rounding() -> None:
    # 0.005 * 65536 = 327.68 → rounds to 328
    assert cal.to_fixed(0.005) == 328


def test_to_fixed_returns_int() -> None:
    result = cal.to_fixed(0.123)
    assert isinstance(result, int)


def test_to_fixed_small_value() -> None:
    result = cal.to_fixed(0.001)
    assert result == round(0.001 * 65536)


def test_to_fixed_large_value() -> None:
    result = cal.to_fixed(10.0)
    assert result == 10 * 65536


# ---------------------------------------------------------------------------
# CalibratedAgriConfig defaults
# ---------------------------------------------------------------------------


def test_calibrated_agri_config_defaults() -> None:
    cfg = cal.CalibratedAgriConfig()
    assert cfg.enabled is True
    assert cfg.spray_radius == 3
    assert cfg.ndvi_scan_radius == 8
    assert cfg.max_growth_stages == 5
    assert 0 < cfg.cropland_density < 1.0
    assert 0 < cfg.pasture_density < 1.0


def test_calibrated_agri_config_disease_rates_positive() -> None:
    cfg = cal.CalibratedAgriConfig()
    assert cfg.disease_spread_rate > 0
    assert cfg.disease_decay_rate > 0


def test_calibrated_agri_config_provenance_zero() -> None:
    cfg = cal.CalibratedAgriConfig()
    assert cfg._plantvillage_samples == 0
    assert cfg._cropnet_counties == 0
    assert cfg._karaagroai_images == 0


# ---------------------------------------------------------------------------
# analyse_plantvillage (dry-run)
# ---------------------------------------------------------------------------


def test_plantvillage_dry_run_returns_dict(tmp_path: Path) -> None:
    result = cal.analyse_plantvillage(tmp_path, dry_run=True)
    assert isinstance(result, dict)


def test_plantvillage_dry_run_keys(tmp_path: Path) -> None:
    result = cal.analyse_plantvillage(tmp_path, dry_run=True)
    for key in ("disease_spread_rate", "disease_decay_rate", "initial_crop_health", "samples"):
        assert key in result, f"Missing key: {key}"


def test_plantvillage_dry_run_samples_zero(tmp_path: Path) -> None:
    result = cal.analyse_plantvillage(tmp_path, dry_run=True)
    assert result["samples"] == 0


def test_plantvillage_dry_run_fixed_point_values(tmp_path: Path) -> None:
    result = cal.analyse_plantvillage(tmp_path, dry_run=True)
    assert result["disease_spread_rate"] == cal.to_fixed(0.005)
    assert result["disease_decay_rate"] == cal.to_fixed(0.002)
    assert result["initial_crop_health"] == cal.to_fixed(0.9)


def test_plantvillage_dry_run_positive_rates(tmp_path: Path) -> None:
    result = cal.analyse_plantvillage(tmp_path, dry_run=True)
    assert result["disease_spread_rate"] > 0
    assert result["disease_decay_rate"] > 0


# ---------------------------------------------------------------------------
# analyse_cropnet (dry-run)
# ---------------------------------------------------------------------------


def test_cropnet_dry_run_returns_dict(tmp_path: Path) -> None:
    result = cal.analyse_cropnet(tmp_path, dry_run=True)
    assert isinstance(result, dict)


def test_cropnet_dry_run_keys(tmp_path: Path) -> None:
    result = cal.analyse_cropnet(tmp_path, dry_run=True)
    for key in ("ndvi_scan_radius", "thermal_scan_radius", "counties"):
        assert key in result, f"Missing key: {key}"


def test_cropnet_dry_run_radii_positive(tmp_path: Path) -> None:
    result = cal.analyse_cropnet(tmp_path, dry_run=True)
    assert result["ndvi_scan_radius"] > 0
    assert result["thermal_scan_radius"] > 0


def test_cropnet_dry_run_counties_zero(tmp_path: Path) -> None:
    result = cal.analyse_cropnet(tmp_path, dry_run=True)
    assert result["counties"] == 0


def test_cropnet_dry_run_ndvi_larger_than_thermal(tmp_path: Path) -> None:
    result = cal.analyse_cropnet(tmp_path, dry_run=True)
    assert result["ndvi_scan_radius"] >= result["thermal_scan_radius"]


# ---------------------------------------------------------------------------
# analyse_karaagroai (dry-run)
# ---------------------------------------------------------------------------


def test_karaagroai_dry_run_returns_dict(tmp_path: Path) -> None:
    result = cal.analyse_karaagroai(tmp_path, dry_run=True)
    assert isinstance(result, dict)


def test_karaagroai_dry_run_keys(tmp_path: Path) -> None:
    result = cal.analyse_karaagroai(tmp_path, dry_run=True)
    for key in ("spray_radius", "spray_efficacy", "images"):
        assert key in result, f"Missing key: {key}"


def test_karaagroai_dry_run_images_zero(tmp_path: Path) -> None:
    result = cal.analyse_karaagroai(tmp_path, dry_run=True)
    assert result["images"] == 0


def test_karaagroai_dry_run_spray_radius_positive(tmp_path: Path) -> None:
    result = cal.analyse_karaagroai(tmp_path, dry_run=True)
    assert result["spray_radius"] > 0


def test_karaagroai_dry_run_efficacy_positive(tmp_path: Path) -> None:
    result = cal.analyse_karaagroai(tmp_path, dry_run=True)
    assert result["spray_efficacy"] > 0


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


def _dry_results(tmp_path: Path) -> tuple[dict, dict, dict]:
    pv = cal.analyse_plantvillage(tmp_path, dry_run=True)
    cn = cal.analyse_cropnet(tmp_path, dry_run=True)
    ka = cal.analyse_karaagroai(tmp_path, dry_run=True)
    return pv, cn, ka


def test_build_config_returns_instance(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    assert isinstance(cfg, cal.CalibratedAgriConfig)


def test_build_config_merges_disease_rates(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    assert cfg.disease_spread_rate == pv["disease_spread_rate"]
    assert cfg.disease_decay_rate == pv["disease_decay_rate"]


def test_build_config_merges_scan_radii(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    assert cfg.ndvi_scan_radius == cn["ndvi_scan_radius"]
    assert cfg.thermal_scan_radius == cn["thermal_scan_radius"]


def test_build_config_merges_spray(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    assert cfg.spray_radius == ka["spray_radius"]
    assert cfg.spray_efficacy == ka["spray_efficacy"]


def test_build_config_provenance_populated(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    assert cfg._plantvillage_samples == pv.get("samples", 0)
    assert cfg._cropnet_counties == cn.get("counties", 0)
    assert cfg._karaagroai_images == ka.get("images", 0)


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------


def test_export_json_creates_file(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "output.json"
    cal.export_json(cfg, out)
    assert out.exists()


def test_export_json_valid_structure(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "output.json"
    cal.export_json(cfg, out)
    data = json.loads(out.read_text())
    assert "agri" in data
    assert "_provenance" in data


def test_export_json_agri_fields(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "output.json"
    cal.export_json(cfg, out)
    agri = json.loads(out.read_text())["agri"]
    assert "disease_spread_rate" in agri
    assert "spray_radius" in agri
    assert "ndvi_scan_radius" in agri
    assert "enabled" in agri


def test_export_json_no_private_fields_in_agri(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "output.json"
    cal.export_json(cfg, out)
    agri = json.loads(out.read_text())["agri"]
    private_keys = [k for k in agri if k.startswith("_")]
    assert private_keys == [], f"Private fields in agri: {private_keys}"


def test_export_json_provenance_has_source(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "output.json"
    cal.export_json(cfg, out)
    prov = json.loads(out.read_text())["_provenance"]
    assert "source" in prov
    assert "plantvillage_samples" in prov
    assert "cropnet_counties" in prov
    assert "karaagroai_images" in prov


def test_export_json_creates_parent_dirs(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "nested" / "dirs" / "config.json"
    cal.export_json(cfg, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# export_toml (requires tomli_w — skip if unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not cal._TOMLI_W_AVAILABLE, reason="tomli_w not installed")
def test_export_toml_creates_file(tmp_path: Path) -> None:
    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "agri.toml"
    cal.export_toml(cfg, out)
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not cal._TOMLI_W_AVAILABLE, reason="tomli_w not installed")
def test_export_toml_readable(tmp_path: Path) -> None:
    try:
        import tomllib  # noqa: PLC0415
    except ImportError:
        pytest.skip("tomllib not available for reading")

    pv, cn, ka = _dry_results(tmp_path)
    cfg = cal.build_config(pv, cn, ka)
    out = tmp_path / "agri.toml"
    cal.export_toml(cfg, out)
    with out.open("rb") as f:
        data = tomllib.load(f)
    assert "agri" in data
    assert "disease_spread_rate" in data["agri"]


def test_export_toml_exits_without_tomli_w(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cal, "_TOMLI_W_AVAILABLE", False)
    with pytest.raises(SystemExit):
        cal.export_toml(cal.CalibratedAgriConfig(), tmp_path / "out.toml")


# ---------------------------------------------------------------------------
# Fallback behaviour when optional packages are unavailable
# ---------------------------------------------------------------------------


def test_plantvillage_falls_back_on_missing_package(tmp_path: Path) -> None:
    """analyse_plantvillage returns defaults when datasets package is absent."""
    original = cal._DATASETS_AVAILABLE
    cal._DATASETS_AVAILABLE = False
    try:
        result = cal.analyse_plantvillage(tmp_path, dry_run=False)
    finally:
        cal._DATASETS_AVAILABLE = original
    assert result["disease_spread_rate"] == cal.to_fixed(0.005)
    assert result["samples"] == 0


def test_cropnet_falls_back_on_missing_package(tmp_path: Path) -> None:
    """analyse_cropnet returns defaults when huggingface_hub is absent."""
    original = cal._HF_HUB_AVAILABLE
    cal._HF_HUB_AVAILABLE = False
    try:
        result = cal.analyse_cropnet(tmp_path, dry_run=False)
    finally:
        cal._HF_HUB_AVAILABLE = original
    assert result["ndvi_scan_radius"] == 8
    assert result["counties"] == 0


def test_karaagroai_falls_back_on_missing_package(tmp_path: Path) -> None:
    """analyse_karaagroai returns defaults when huggingface_hub is absent."""
    original = cal._HF_HUB_AVAILABLE
    cal._HF_HUB_AVAILABLE = False
    try:
        result = cal.analyse_karaagroai(tmp_path, dry_run=False)
    finally:
        cal._HF_HUB_AVAILABLE = original
    assert result["spray_radius"] == 3
    assert result["images"] == 0


# ---------------------------------------------------------------------------
# Live-path coverage using mocks
# ---------------------------------------------------------------------------


def test_plantvillage_live_path_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the actual data-processing code path when datasets is mocked."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    # 60 diseased (labels 0-1), 40 healthy (label 2)
    mock_ds.__getitem__.side_effect = lambda i: {"label": i % 3}
    label_feature = MagicMock()
    label_feature.names = ["disease_a", "disease_b", "healthy"]
    mock_ds.features = {"label": label_feature}

    monkeypatch.setattr(cal, "_DATASETS_AVAILABLE", True)
    monkeypatch.setattr(cal, "_load_dataset", lambda *_args, **_kwargs: mock_ds)

    result = cal.analyse_plantvillage(tmp_path, dry_run=False)
    assert result["disease_spread_rate"] > 0
    assert result["disease_decay_rate"] > 0
    assert result["samples"] == 100


def test_plantvillage_live_path_exception_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions in the live path fall back to defaults."""
    monkeypatch.setattr(cal, "_DATASETS_AVAILABLE", True)
    monkeypatch.setattr(cal, "_load_dataset", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))

    result = cal.analyse_plantvillage(tmp_path, dry_run=False)
    assert result["disease_spread_rate"] == cal.to_fixed(0.005)


def test_cropnet_live_path_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CropNet live path when hf_hub_download succeeds."""
    monkeypatch.setattr(cal, "_HF_HUB_AVAILABLE", True)
    monkeypatch.setattr(cal, "_hf_hub_download", lambda **_kwargs: "/fake/path/README.md")

    result = cal.analyse_cropnet(tmp_path, dry_run=False)
    assert result["ndvi_scan_radius"] == 8
    assert result["counties"] == 2200


def test_cropnet_live_path_exception_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal, "_HF_HUB_AVAILABLE", True)
    monkeypatch.setattr(cal, "_hf_hub_download", lambda **_kwargs: (_ for _ in ()).throw(OSError("no network")))

    result = cal.analyse_cropnet(tmp_path, dry_run=False)
    assert result["ndvi_scan_radius"] == 8
    assert result["counties"] == 0


def test_karaagroai_live_path_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test KaraAgroAI live path when hf_hub_download succeeds."""
    monkeypatch.setattr(cal, "_HF_HUB_AVAILABLE", True)
    monkeypatch.setattr(cal, "_hf_hub_download", lambda **_kwargs: "/fake/path/README.md")

    result = cal.analyse_karaagroai(tmp_path, dry_run=False)
    assert result["spray_radius"] == 3
    assert result["images"] == 8784


def test_karaagroai_live_path_exception_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal, "_HF_HUB_AVAILABLE", True)
    monkeypatch.setattr(cal, "_hf_hub_download", lambda **_kwargs: (_ for _ in ()).throw(OSError("timeout")))

    result = cal.analyse_karaagroai(tmp_path, dry_run=False)
    assert result["spray_radius"] == 3
    assert result["images"] == 0


# ---------------------------------------------------------------------------
# main() CLI coverage
# ---------------------------------------------------------------------------


def test_main_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() runs end-to-end in dry-run mode (no downloads) and produces JSON output."""
    out_toml = tmp_path / "agri.toml"
    out_json = tmp_path / "agri.json"

    # Patch sys.argv and export_toml (requires tomli_w which may not be installed)
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate_agri.py", "--dry-run", "--output", str(out_toml), "--cache-dir", str(tmp_path)],
    )
    # Mock export_toml to avoid tomli_w dependency
    monkeypatch.setattr(cal, "export_toml", lambda cfg, path: None)

    cal.main()

    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert "agri" in data
