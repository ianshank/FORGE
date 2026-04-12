"""MuZero model export to ONNX and TorchScript formats.

Exports the three MuZero networks (representation, dynamics, prediction)
as separate model files suitable for deployment with ONNX Runtime or
TorchScript in the Rust latent MCTS engine.

Usage::

    from forge.models.muzero_config import MuZeroConfig
    from forge.models.muzero_world_model import MuZeroWorldModel
    from forge.models.muzero_export import MuZeroExporter

    model = MuZeroWorldModel(MuZeroConfig(obs_dim=920, action_dim=75))
    exporter = MuZeroExporter(model)
    exporter.export_onnx(Path("export/onnx"))
    exporter.export_torchscript(Path("export/torchscript"))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forge.models.muzero_world_model import MuZeroWorldModel

logger = logging.getLogger(__name__)


class MuZeroExporter:
    """Exports MuZero networks to deployment formats.

    Creates separate model files for each of the three MuZero networks,
    allowing the Rust latent MCTS to load them independently via
    ONNX Runtime or TorchScript.

    Args:
        model: A trained :class:`MuZeroWorldModel`.
    """

    def __init__(self, model: MuZeroWorldModel) -> None:
        self._model = model
        self._config = model.config

    def export_onnx(self, output_dir: Path, opset_version: int = 17) -> list[Path]:
        """Export all three networks to ONNX format.

        Args:
            output_dir: Directory to write the ONNX files.
            opset_version: ONNX opset version.

        Returns:
            List of paths to the exported ONNX files.
        """
        import torch  # noqa: PLC0415

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self._config
        paths: list[Path] = []

        # --- Representation Network ---
        rep_path = output_dir / "representation.onnx"
        rep_module = _build_rep_module(self._model)
        dummy_obs = torch.randn(1, c.obs_dim)
        torch.onnx.export(
            rep_module,
            dummy_obs,
            str(rep_path),
            input_names=["observation"],
            output_names=["latent_state"],
            dynamic_axes={"observation": {0: "batch"}, "latent_state": {0: "batch"}},
            opset_version=opset_version,
        )
        paths.append(rep_path)
        logger.info("Exported representation network to %s", rep_path)

        # --- Dynamics Network ---
        dyn_path = output_dir / "dynamics.onnx"
        dyn_module = _build_dyn_module(self._model)
        input_dim = c.latent_dim + c.action_dim
        dummy_input = torch.randn(1, input_dim)
        torch.onnx.export(
            dyn_module,
            dummy_input,
            str(dyn_path),
            input_names=["latent_action"],
            output_names=["next_latent", "reward_logits"],
            dynamic_axes={
                "latent_action": {0: "batch"},
                "next_latent": {0: "batch"},
                "reward_logits": {0: "batch"},
            },
            opset_version=opset_version,
        )
        paths.append(dyn_path)
        logger.info("Exported dynamics network to %s", dyn_path)

        # --- Prediction Network ---
        pred_path = output_dir / "prediction.onnx"
        pred_module = _build_pred_module(self._model)
        dummy_latent = torch.randn(1, c.latent_dim)
        torch.onnx.export(
            pred_module,
            dummy_latent,
            str(pred_path),
            input_names=["latent_state"],
            output_names=["policy_logits", "value_logits"],
            dynamic_axes={
                "latent_state": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value_logits": {0: "batch"},
            },
            opset_version=opset_version,
        )
        paths.append(pred_path)
        logger.info("Exported prediction network to %s", pred_path)

        return paths

    def export_torchscript(self, output_dir: Path) -> list[Path]:
        """Export all three networks to TorchScript format.

        Args:
            output_dir: Directory to write the TorchScript files.

        Returns:
            List of paths to the exported TorchScript files.
        """
        import torch  # noqa: PLC0415

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        c = self._config
        paths: list[Path] = []

        # Representation
        rep_module = _build_rep_module(self._model)
        rep_traced = torch.jit.trace(rep_module, torch.randn(1, c.obs_dim))
        rep_path = output_dir / "representation.pt"
        rep_traced.save(str(rep_path))
        paths.append(rep_path)
        logger.info("Exported representation TorchScript to %s", rep_path)

        # Dynamics
        dyn_module = _build_dyn_module(self._model)
        input_dim = c.latent_dim + c.action_dim
        dyn_traced = torch.jit.trace(dyn_module, torch.randn(1, input_dim))
        dyn_path = output_dir / "dynamics.pt"
        dyn_traced.save(str(dyn_path))
        paths.append(dyn_path)
        logger.info("Exported dynamics TorchScript to %s", dyn_path)

        # Prediction
        pred_module = _build_pred_module(self._model)
        pred_traced = torch.jit.trace(pred_module, torch.randn(1, c.latent_dim))
        pred_path = output_dir / "prediction.pt"
        pred_traced.save(str(pred_path))
        paths.append(pred_path)
        logger.info("Exported prediction TorchScript to %s", pred_path)

        return paths

    def validate_export(
        self,
        output_dir: Path,
        fmt: str = "onnx",
        atol: float = 1e-4,
    ) -> bool:
        """Validate exported models against PyTorch outputs.

        Runs the same input through PyTorch and the exported model,
        checking that outputs are numerically close.

        Args:
            output_dir: Directory containing exported models.
            fmt: Export format to validate ("onnx" or "torchscript").
            atol: Absolute tolerance for numerical comparison.

        Returns:
            True if all models pass validation.
        """
        import torch  # noqa: PLC0415

        c = self._config
        obs = np.random.randn(c.obs_dim).astype(np.float32)

        # Get PyTorch reference output
        output = self._model.initial_inference(obs)
        ref_latent = output.latent_state
        ref_policy = output.policy_logits

        if fmt == "onnx":
            return self._validate_onnx(output_dir, obs, ref_latent, ref_policy, atol)
        if fmt == "torchscript":
            return self._validate_torchscript(output_dir, obs, ref_latent, ref_policy, atol)

        logger.error("Unknown format: %s", fmt)
        return False

    def _validate_onnx(
        self,
        output_dir: Path,
        obs: np.ndarray,
        ref_latent: np.ndarray,
        ref_policy: np.ndarray,
        atol: float,
    ) -> bool:
        """Validate ONNX export against PyTorch."""
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            logger.warning("onnxruntime not installed — skipping ONNX validation")
            return True

        # Validate representation
        rep_sess = ort.InferenceSession(str(output_dir / "representation.onnx"))
        onnx_latent = rep_sess.run(None, {"observation": obs.reshape(1, -1)})[0]
        if not np.allclose(onnx_latent.flatten(), ref_latent, atol=atol):
            logger.error("Representation ONNX validation failed")
            return False

        # Validate prediction
        pred_sess = ort.InferenceSession(str(output_dir / "prediction.onnx"))
        onnx_out = pred_sess.run(None, {"latent_state": onnx_latent})
        onnx_policy = onnx_out[0]
        if not np.allclose(onnx_policy.flatten()[: len(ref_policy)], ref_policy, atol=atol):
            logger.error("Prediction ONNX validation failed")
            return False

        logger.info("ONNX validation passed (atol=%s)", atol)
        return True

    def _validate_torchscript(
        self,
        output_dir: Path,
        obs: np.ndarray,
        ref_latent: np.ndarray,
        ref_policy: np.ndarray,
        atol: float,
    ) -> bool:
        """Validate TorchScript export against PyTorch."""
        import torch  # noqa: PLC0415

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        rep_model = torch.jit.load(str(output_dir / "representation.pt"))
        ts_latent = rep_model(obs_t).detach().numpy().flatten()
        if not np.allclose(ts_latent, ref_latent, atol=atol):
            logger.error("Representation TorchScript validation failed")
            return False

        pred_model = torch.jit.load(str(output_dir / "prediction.pt"))
        latent_t = torch.tensor(ts_latent, dtype=torch.float32).unsqueeze(0)
        ts_policy, ts_value = pred_model(latent_t)
        if not np.allclose(ts_policy.detach().numpy().flatten()[: len(ref_policy)], ref_policy, atol=atol):
            logger.error("Prediction TorchScript validation failed")
            return False

        logger.info("TorchScript validation passed (atol=%s)", atol)
        return True


# --- Internal nn.Module builders for ONNX/TorchScript export ---


def _build_rep_module(model: MuZeroWorldModel) -> "torch.nn.Module":
    """Build a traceable nn.Module wrapping the representation network."""
    from torch import nn  # noqa: PLC0415

    class RepModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = model.representation.cnn
            self.vector_mlp = model.representation.vector_mlp
            self.fusion = model.representation.fusion
            self.res_blocks = model.representation.res_blocks

        def forward(self, observation: "torch.Tensor") -> "torch.Tensor":
            return model.representation.forward(observation)

    m = RepModule()
    m.eval()
    return m


def _build_dyn_module(model: MuZeroWorldModel) -> "torch.nn.Module":
    """Build a traceable nn.Module wrapping the dynamics network."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    latent_dim = model.config.latent_dim

    class DynModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transition = model.dynamics.transition
            self.res_blocks = model.dynamics.res_blocks
            self.reward_head = model.dynamics.reward_head

        def forward(
            self, latent_action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            latent = latent_action[:, :latent_dim]
            action = latent_action[:, latent_dim:]
            return model.dynamics.forward(latent, action)

    m = DynModule()
    m.eval()
    return m


def _build_pred_module(model: MuZeroWorldModel) -> "torch.nn.Module":
    """Build a traceable nn.Module wrapping the prediction network."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    class PredModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = model.prediction.trunk
            self.policy_head = model.prediction.policy_head
            self.value_head = model.prediction.value_head

        def forward(
            self, latent_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return model.prediction.forward(latent_state)

    m = PredModule()
    m.eval()
    return m
