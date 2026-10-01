"""Engine service managing model lifecycle, thread-safe inference, and dispatch."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, Union

from ..config import settings
from ..core.engine import NirnayaEngine
from ..core.model_registry import registry, get_hardware_info
from ..core.schema import Prediction, Question, parse_questions


class EngineService:
    """Thread-safe singleton managing Nirnaya inference engines."""

    def __init__(self) -> None:
        self._engines: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_engine(self, model_name: Optional[str] = None) -> Any:
        """Returns or instantiates the engine for the requested model."""
        target_model = model_name or settings.DEFAULT_MODEL
        with self._lock:
            if target_model not in self._engines:
                print(f"[EngineService] Initializing engine for model '{target_model}'...")
                # Instantiate NirnayaEngine (handles transformers, llama_cpp, prism_llama)
                engine = NirnayaEngine(model_name=target_model)
                self._engines[target_model] = engine
                print(f"[EngineService] Successfully loaded '{target_model}'.")
            return self._engines[target_model]

    def predict(
        self,
        state: Union[str, Dict[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        model_name: Optional[str] = None,
    ) -> Prediction:
        """Executes single next-token readout across typed questions against state."""
        target_model = model_name or settings.DEFAULT_MODEL
        engine = self.get_engine(target_model)

        # Normalize state
        if isinstance(state, dict):
            state_text = "\n".join(f"{k}: {v}" for k, v in state.items())
        elif not isinstance(state, str):
            state_text = json.dumps(state)
        else:
            state_text = state

        # Parse questions using Nirnaya schema parser
        parsed_questions = parse_questions(questions)

        # Execute decision engine
        prediction = engine.predict(state=state_text, questions=parsed_questions)
        prediction.model = target_model
        return prediction

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns list of registered models and their status."""
        models = registry.list_models()
        for m in models:
            m["is_default"] = (m["alias"] == settings.DEFAULT_MODEL)
            m["is_loaded"] = (m["alias"] in self._engines)
        return models

    def get_hardware_status(self) -> Dict[str, Any]:
        return get_hardware_info()

    def warmup(self) -> None:
        """Warms up default model with a lightweight test pass."""
        print(f"[EngineService] Warming up default model '{settings.DEFAULT_MODEL}'...")
        t0 = time.perf_counter()
        sample_q = {
            "test_q": {
                "type": "noul",
                "instructions": "Is system operational?"
            }
        }
        self.predict(state="System test state", questions=sample_q, model_name=settings.DEFAULT_MODEL)
        dur = (time.perf_counter() - t0) * 1000.0
        print(f"[EngineService] Warmup completed in {dur:.1f}ms.")


# Global singleton instance
engine_service = EngineService()
