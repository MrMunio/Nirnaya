"""Model Registry & Hardware-Aware Dispatcher for Nirnaya.

Loads configuration from models_config.json, automatically maps aliases
to engine backends, resolves local paths or downloads missing weights,
and adapts hardware settings (CUDA vs CPU) seamlessly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CONFIG_PATH = Path(__file__).parent / "models_config.json"


def is_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_hardware_info() -> Dict[str, Any]:
    cuda_avail = is_cuda_available()
    device = "cuda" if cuda_avail else "cpu"
    n_gpu_layers = 99 if cuda_avail else 0
    return {
        "cuda_available": cuda_avail,
        "recommended_device": device,
        "recommended_ngl": n_gpu_layers,
    }


class ModelRegistry:
    def __init__(self, config_path: Optional[Path | str] = None):
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {"models": {}}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for alias, item in self.data.get("models", {}).items():
            models.append({
                "alias": alias,
                "name": item.get("name", alias),
                "backend": item.get("backend", "transformers"),
                "description": item.get("description", ""),
                "enable_thinking": item.get("enable_thinking", False),
            })
        return models

    def resolve(self, model_spec: str, user_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolves a model spec (alias, path, or HF repo) into complete execution parameters."""
        overrides = dict(user_overrides or {})
        models = self.data.get("models", {})

        matched_key = None
        matched_cfg = None

        # 1. Exact alias match
        if model_spec in models:
            matched_key = model_spec
            matched_cfg = dict(models[model_spec])
        else:
            # 2. Match by model_id, model_path, or hf_repo
            spec_lower = model_spec.lower().replace("\\", "/")
            for k, v in models.items():
                m_id = str(v.get("model_id", "")).lower()
                m_path = str(v.get("model_path", "")).lower().replace("\\", "/")
                hf_repo = str(v.get("hf_repo", "")).lower()
                if spec_lower in (m_id, m_path, hf_repo) or spec_lower.endswith(k):
                    matched_key = k
                    matched_cfg = dict(v)
                    break

        if matched_cfg is None:
            # Fallback heuristic for ad-hoc / unconfigured models
            if model_spec.endswith(".gguf") or "bonsai" in model_spec.lower() or "prism" in model_spec.lower():
                backend = "prism_llama"
            elif "gguf" in model_spec.lower():
                backend = "llama_cpp"
            else:
                backend = "transformers"
            matched_cfg = {
                "name": model_spec,
                "backend": backend,
                "model_id": model_spec,
                "model_path": model_spec,
            }

        # Apply hardware auto-detection
        hw = get_hardware_info()
        if matched_cfg.get("device") == "auto":
            matched_cfg["device"] = hw["recommended_device"]
        if matched_cfg.get("n_gpu_layers") == "auto":
            matched_cfg["n_gpu_layers"] = hw["recommended_ngl"]

        # Apply user overrides (skip 'auto' so config defaults are preserved)
        for k, v in overrides.items():
            if v is not None and v != "auto":
                matched_cfg[k] = v

        # Resolve local paths / trigger download if needed
        self._ensure_artifacts(matched_cfg)

        return matched_cfg

    def _ensure_artifacts(self, cfg: Dict[str, Any]) -> None:
        """Verifies model files exist locally or downloads them via huggingface_hub."""
        candidate_roots = [
            Path.cwd(),
            Path(__file__).resolve().parent.parent.parent.parent,
            Path(__file__).resolve().parent.parent.parent,
            Path(__file__).resolve().parent.parent,
        ]
        backend = cfg.get("backend")

        if backend in ("prism_llama", "llama_cpp"):
            model_path_str = cfg.get("model_path")
            local_dir_str = cfg.get("local_dir")
            hf_repo = cfg.get("hf_repo")
            filename = cfg.get("filename")

            # Check if direct file exists
            if model_path_str:
                p = Path(model_path_str)
                if p.is_absolute() and p.exists():
                    cfg["resolved_model_path"] = str(p)
                    return
                for root in candidate_roots:
                    cand = root / p
                    if cand.exists():
                        cfg["resolved_model_path"] = str(cand.resolve())
                        return

            # Check local_dir + filename
            if local_dir_str and filename:
                for root in candidate_roots:
                    cand_file = root / local_dir_str / filename
                    if cand_file.exists():
                        cfg["resolved_model_path"] = str(cand_file.resolve())
                        return

                target_dir = candidate_roots[0] / local_dir_str

                # Download needed
                if hf_repo:
                    print(f"[ModelRegistry] Model file '{filename}' not found locally.")
                    print(f"[ModelRegistry] Downloading from {hf_repo} -> {target_dir} ...")
                    target_dir.mkdir(parents=True, exist_ok=True)
                    from huggingface_hub import hf_hub_download
                    downloaded = hf_hub_download(
                        repo_id=hf_repo,
                        filename=filename,
                        local_dir=str(target_dir),
                        local_dir_use_symlinks=False,
                    )
                    cfg["resolved_model_path"] = downloaded
                    print(f"[ModelRegistry] Download complete: {downloaded}")
                    return

            # If fallback model_path was provided and exists
            if model_path_str and Path(model_path_str).exists():
                cfg["resolved_model_path"] = str(Path(model_path_str).resolve())
            elif model_path_str:
                cfg["resolved_model_path"] = model_path_str
        else:
            # Transformers model_id
            cfg["resolved_model_path"] = cfg.get("model_id") or cfg.get("model_path")


# Singleton instance
registry = ModelRegistry()
