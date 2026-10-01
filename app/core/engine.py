"""NirnayaEngine - single-pass typed decisions from a small causal LM (no generation, no training).

Per request:
  1. STATIC prefix (system prompt + few-shot) is prefilled once per engine and cached.
  2. STATE is prefilled once per request on a copy of the static cache.
  3. Each question is a short suffix run on a copy of the state cache; we read ONE next-token
     distribution and restrict it to the valid label tokens.
Choice questions are asked under several option orders (cyclic rotations) and pooled to cancel
position bias; disagreement between orders is reported as `agreement`.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from .labels import LabelSpace, build_label_space
from .prompting import PromptBuilder, render_question
from .readout import build_answer, label_logprobs, pool_logps
from .schema import Answer, ChoiceQuestion, NoulQuestion, Prediction, Question, parse_questions


class NirnayaEngine:
    def __new__(cls, *args, **kwargs):
        if cls is not NirnayaEngine:
            return super().__new__(cls)

        # If an instantiated PyTorch model object is passed directly, use standard class
        if kwargs.get("model") is not None or (len(args) > 0 and not isinstance(args[0], str)):
            return super().__new__(cls)

        from .model_registry import registry
        model_name = kwargs.get("model_name")
        if not model_name and len(args) >= 3 and isinstance(args[2], str):
            model_name = args[2]
        if not model_name and len(args) == 1 and isinstance(args[0], str):
            model_name = args[0]
        if not isinstance(model_name, str):
            model_name = kwargs.get("model", "qwen2.5-0.5b")
        if not isinstance(model_name, str):
            return super().__new__(cls)

        # Resolve config via registry
        cfg = registry.resolve(model_name, kwargs)
        backend = kwargs.get("backend", "auto")
        if backend == "auto":
            backend = cfg.get("backend", "transformers")

        # Merge resolved cfg with explicit kwargs
        engine_kwargs = dict(cfg)
        for k, v in kwargs.items():
            if v is not None and v != "auto":
                engine_kwargs[k] = v

        model_path = cfg.get("resolved_model_path", model_name)
        engine_kwargs.pop("backend", None)
        engine_kwargs.pop("model_name", None)
        engine_kwargs.pop("model_path", None)

        if backend == "llama_cpp":
            try:
                import llama_cpp
                from .llama_engine import LlamaCppEngine
                return LlamaCppEngine(
                    model_path=model_path,
                    **engine_kwargs,
                )
            except ImportError:
                print("[NirnayaEngine] 'llama_cpp' package not installed; falling back to CUDA 'prism_llama' (llama-server.exe).")
                backend = "prism_llama"

        if backend == "prism_llama":
            from .prism_engine import PrismLlamaEngine
            return PrismLlamaEngine(
                model_name=model_path,
                **engine_kwargs,
            )

        return super().__new__(cls)

    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = "auto",
        dtype: str = "float32",
        n_perms: int = 3,
        few_shot: bool = True,
        use_chat_template: Optional[bool] = None,
        enable_thinking: bool = False,
        calibrator=None,
        num_threads: Optional[int] = None,
        backend: str = "auto",
        dedicated_prompts: bool = True,
        **kwargs,
    ):
        import warnings
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
        from .model_registry import registry

        self._torch = torch
        self._DynamicCache = DynamicCache
        if num_threads:
            torch.set_num_threads(num_threads)

        cfg = registry.resolve(model_name, kwargs) if isinstance(model_name, str) else {}
        resolved_model = cfg.get("resolved_model_path") or cfg.get("model_id") or (model_name if isinstance(model_name, str) else getattr(model, "name_or_path", "custom_model"))

        resolved_device = device or "auto"
        if resolved_device == "auto":
            resolved_device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        elif str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"Device '{resolved_device}' requested but CUDA is not available. Falling back to CPU.",
                UserWarning,
            )
            resolved_device = "cpu"

        if model is None or tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(resolved_model)
            model_dtype = dtype if dtype != "auto" else cfg.get("dtype", "float32")
            torch_dtype = getattr(torch, model_dtype) if hasattr(torch, model_dtype) else torch.float32
            model = AutoModelForCausalLM.from_pretrained(resolved_model, torch_dtype=torch_dtype)
        self.model_name = resolved_model
        self.tok = tokenizer
        self.model = model.to(resolved_device).eval()
        self.device = resolved_device
        resolved_perms = kwargs.get("n_perms") or n_perms or cfg.get("n_perms", 1)
        self.n_perms = max(1, resolved_perms)
        self.calibrator = calibrator
        self.dedicated_prompts = cfg.get("dedicated_prompts", dedicated_prompts)

        if dedicated_prompts:
            self.prompts: Dict[str, PromptBuilder] = {
                "choice": PromptBuilder(
                    tokenizer,
                    use_chat_template=use_chat_template,
                    few_shot=few_shot,
                    answer_cue="Answer:",
                    enable_thinking=enable_thinking,
                    qtype="choice",
                ),
                "score": PromptBuilder(
                    tokenizer,
                    use_chat_template=use_chat_template,
                    few_shot=few_shot,
                    answer_cue="Answer:",
                    enable_thinking=enable_thinking,
                    qtype="score",
                ),
                "noul": PromptBuilder(
                    tokenizer,
                    use_chat_template=use_chat_template,
                    few_shot=few_shot,
                    answer_cue="Answer:",
                    enable_thinking=enable_thinking,
                    qtype="noul",
                ),
            }
            self.static_caches: Dict[str, Any] = {}
            for qt, pb in self.prompts.items():
                self._forward(pb.static_ids(), None)
                self.static_caches[qt] = self._last_cache
        else:
            self.prompts = {}
            self.static_caches = {}

        self.prompt = PromptBuilder(
            tokenizer,
            use_chat_template=use_chat_template,
            few_shot=few_shot,
            enable_thinking=enable_thinking,
        )
        self._spaces: Dict[tuple, LabelSpace] = {}

        # static prefix: computed once, reused (via copies) for every request (or general fallback)
        self.static_ids = self.prompt.static_ids()
        self._forward(self.static_ids, None)
        self.static_cache = self._last_cache

    # ------------------------------------------------------------------ low level
    def _forward(self, ids: List[int], cache=None):
        """Run `ids` through the decoder body, extending `cache` in place. Returns last hidden state."""
        with self._torch.inference_mode():
            t = self._torch.tensor([ids], device=self.device)
            out = self.model.base_model(input_ids=t, past_key_values=cache, use_cache=True)
            self._last_cache = out.past_key_values
            return out.last_hidden_state[0, -1]

    def _logits(self, hidden):
        with self._torch.inference_mode():
            return self.model.get_output_embeddings()(hidden)

    def _space(self, qtype: str, labels: List[str]) -> LabelSpace:
        key = (qtype, tuple(labels))
        if key not in self._spaces:
            self._spaces[key] = build_label_space(self.tok, labels, case_variants=(qtype == "noul"))
        return self._spaces[key]

    def _orders(self, q: Question) -> List[Optional[List[str]]]:
        if not isinstance(q, ChoiceQuestion):
            return [None]
        keys, n = list(q.options), len(q.options)
        k = min(self.n_perms, n)
        offsets = sorted({round(i * n / k) % n for i in range(k)})
        return [keys[o:] + keys[:o] for o in offsets]

    # ------------------------------------------------------------------ one question
    def _answer(self, q: Question, state_cache) -> Answer:
        per_order_logps, valid = [], []
        canonical = list(q.options) if isinstance(q, ChoiceQuestion) else None
        pb = self.prompts.get(q.qtype, self.prompt) if getattr(self, "prompts", None) else self.prompt

        for order in self._orders(q):
            qtext, label_strs = render_question(q, order)
            space = self._space(q.qtype, label_strs)
            cache = copy.deepcopy(state_cache)
            hidden = self._forward(pb.suffix_ids(qtext), cache)
            lp, vm = label_logprobs(self._logits(hidden), space)
            if canonical is not None:                       # map letters back to option keys
                by_key = {key: lp[i] for i, key in enumerate(order)}
                lp = [by_key[key] for key in canonical]
            per_order_logps.append(lp)
            valid.append(vm)

        pooled = pool_logps(per_order_logps)
        final_arg = int(pooled.argmax())
        agreement = sum(int(max(range(len(lp)), key=lp.__getitem__) == final_arg)
                        for lp in per_order_logps) / len(per_order_logps)

        if isinstance(q, ChoiceQuestion):
            labels = canonical
        elif isinstance(q, NoulQuestion):
            labels = ["yes", "no"]
        else:
            labels = [str(i) for i in range(len(q.levels))]
        return build_answer(q, labels, pooled, sum(valid) / len(valid), agreement, self.calibrator)

    # ------------------------------------------------------------------ public API
    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Prediction:
        qs = parse_questions(questions)
        q_map = {q.qid: q for q in qs}
        t0 = time.perf_counter()

        if getattr(self, "dedicated_prompts", False) and getattr(self, "prompts", None):
            qtypes = {q.qtype for q in qs}
            state_caches = {}
            total_state_tokens = 0
            total_static_tokens = 0
            for qt in qtypes:
                pb = self.prompts.get(qt, self.prompt)
                st_ids = pb.state_ids(state)
                base_cache = self.static_caches.get(qt, self.static_cache)
                st_cache = copy.deepcopy(base_cache)
                self._forward(st_ids, st_cache)
                state_caches[qt] = st_cache
                total_state_tokens += len(st_ids)
                total_static_tokens += len(pb.static_ids())
            t1 = time.perf_counter()
            answers = {}
            q_timings = {}
            for q in qs:
                t_q = time.perf_counter()
                answers[q.qid] = self._answer(q, state_caches[q.qtype])
                q_timings[q.qid] = (time.perf_counter() - t_q) * 1e3
        else:
            state_ids = self.prompt.state_ids(state)
            state_cache = copy.deepcopy(self.static_cache)
            self._forward(state_ids, state_cache)
            total_state_tokens = len(state_ids)
            total_static_tokens = len(self.static_ids)
            t1 = time.perf_counter()
            answers = {}
            q_timings = {}
            for q in qs:
                t_q = time.perf_counter()
                answers[q.qid] = self._answer(q, state_cache)
                q_timings[q.qid] = (time.perf_counter() - t_q) * 1e3

        t2 = time.perf_counter()

        timings = {
            "state": (t1 - t0) * 1e3,
            "questions": (t2 - t1) * 1e3,
            "total": (t2 - t0) * 1e3,
            **q_timings,
        }

        return Prediction(
            answers=answers,
            timings_ms=timings,
            tokens={"static_cached": total_static_tokens, "state": total_state_tokens},
            model=getattr(self, "model_name", "nirnaya"),
            questions=q_map,
        )
