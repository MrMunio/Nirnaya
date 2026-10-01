"""LlamaCppEngine - Pure in-process GGUF engine for Nirnaya on CPU & GPU.

Uses llama_cpp.Llama directly for zero-overhead next-token readout,
KV-cache prefix preservation, and exact mathematical logit extraction.
Fully compatible with Nirnaya's permutation pooling, 1D Center-of-Gravity,
and TypeSafe AI Jev contract.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .labels import LabelSpace, build_label_space
from .prompting import PromptBuilder, render_question, render_state
from .readout import build_answer, label_logprobs, pool_logps
from .schema import Answer, ChoiceQuestion, NoulQuestion, Prediction, Question, parse_questions


class LlamaCppTokenizerWrapper:
    """Adapts llama_cpp.Llama tokenizer to Nirnaya's PromptBuilder expectations."""

    def __init__(self, llm_instance: Any, chat_template: Optional[str] = None):
        self.llm = llm_instance
        self._cache: Dict[str, List[int]] = {}
        self.bos_token = "<|im_start|>"

        tmpl = chat_template
        if not tmpl:
            # Check metadata
            try:
                tmpl = self.llm.metadata.get("tokenizer.chat_template")
            except Exception:
                pass
        if not tmpl:
            # Default Qwen/ChatML template
            tmpl = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{'<|im_start|>assistant\n'}}"
                "{% if not enable_thinking %}{{'<think>\n\n</think>\n\n'}}{% endif %}"
                "{% endif %}"
            )
        self.chat_template = tmpl
        import jinja2
        self._jinja_tmpl = jinja2.Environment().from_string(self.chat_template)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text in self._cache:
            return list(self._cache[text])
        raw_b = text.encode("utf-8")
        tokens = self.llm.tokenize(raw_b, add_bos=False, special=True)
        self._cache[text] = tokens
        return list(tokens)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        return self._jinja_tmpl.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )


class LlamaCppEngine:
    """Nirnaya decision engine running on llama_cpp in-process."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        n_perms: int = 3,
        few_shot: bool = True,
        use_chat_template: Optional[bool] = None,
        enable_thinking: bool = False,
        calibrator=None,
        dedicated_prompts: bool = True,
        **kwargs,
    ):
        from llama_cpp import Llama

        resolved_path = os.path.abspath(model_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"GGUF model not found at {resolved_path}")

        self.model_path = resolved_path
        try:
            from ..config import settings
            env_perms = settings.N_PERMS
        except Exception:
            env_val = os.getenv("NIRNAYA_N_PERMS")
            env_perms = int(env_val) if env_val is not None and env_val.strip() != "" else None

        if env_perms is not None:
            self.n_perms = max(1, env_perms)
        else:
            self.n_perms = max(1, n_perms)
        self.calibrator = calibrator
        self.dedicated_prompts = dedicated_prompts
        self._spaces: Dict[tuple, LabelSpace] = {}

        # Resolve CPU threads if not specified
        if n_threads is None:
            n_threads = max(1, os.cpu_count() or 4)

        print(f"[LlamaCppEngine] Loading {self.model_name} (threads={n_threads}, ngl={n_gpu_layers}, ctx={n_ctx}) ...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
        )

        self.tok = LlamaCppTokenizerWrapper(self.llm)
        use_chat = True if use_chat_template is None else use_chat_template

        if dedicated_prompts:
            self.prompts: Dict[str, PromptBuilder] = {
                "choice": PromptBuilder(
                    self.tok,
                    use_chat_template=use_chat,
                    few_shot=few_shot,
                    answer_cue="Answer:",
                    enable_thinking=enable_thinking,
                    qtype="choice",
                ),
                "score": PromptBuilder(
                    self.tok,
                    use_chat_template=use_chat,
                    few_shot=few_shot,
                    answer_cue="Answer:",
                    enable_thinking=enable_thinking,
                    qtype="score",
                ),
                "noul": PromptBuilder(
                    self.tok,
                    use_chat_template=use_chat,
                    few_shot=few_shot,
                    answer_cue="Answer:",
                    enable_thinking=enable_thinking,
                    qtype="noul",
                ),
            }
        else:
            self.prompts = {}

        self.prompt = PromptBuilder(
            self.tok,
            use_chat_template=use_chat,
            few_shot=few_shot,
            answer_cue="Answer:",
            enable_thinking=enable_thinking,
        )

        # Precompute static KV caches once at initialization
        self._static_states: Dict[str, Any] = {}
        if dedicated_prompts:
            for qt, pb in self.prompts.items():
                st_tokens = self.tok.encode(pb._static_text, add_special_tokens=False)
                self.llm.reset()
                self.llm.eval(st_tokens)
                self._static_states[qt] = self.llm.save_state()
        else:
            st_tokens = self.tok.encode(self.prompt._static_text, add_special_tokens=False)
            self.llm.reset()
            self.llm.eval(st_tokens)
            self._static_states["default"] = self.llm.save_state()

        print(f"[LlamaCppEngine] Model and static KV-cache initialized successfully.")

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

    def _answer(self, q: Question, prefix_prompt: str, saved_state: Any = None) -> Answer:
        per_order_logps, valids = [], []
        canonical = list(q.options) if isinstance(q, ChoiceQuestion) else None
        pb = self.prompts.get(q.qtype, self.prompt) if getattr(self, "prompts", None) else self.prompt

        for order in self._orders(q):
            qtext, label_strs = render_question(q, order)
            space = self._space(q.qtype, label_strs)
            suffix = qtext + pb._tail + pb.answer_cue

            import llama_cpp
            if saved_state is not None:
                self.llm.load_state(saved_state)
                tokens = self.tok.encode(suffix, add_special_tokens=False)
                self.llm.eval(tokens)
            else:
                full_prompt = prefix_prompt + suffix
                tokens = self.tok.encode(full_prompt, add_special_tokens=False)
                self.llm.reset()
                self.llm.eval(tokens)

            ptr = llama_cpp.llama_get_logits(self.llm.ctx)
            arr = np.ctypeslib.as_array(ptr, shape=(self.llm.n_vocab(),)).copy()

            log_probs, valid_mass = label_logprobs(arr, space)

            if canonical is not None:
                by_key = {key: log_probs[i] for i, key in enumerate(order)}
                log_probs = [by_key[key] for key in canonical]

            per_order_logps.append(log_probs)
            valids.append(valid_mass)

        pooled = pool_logps(per_order_logps)
        final_arg = int(pooled.argmax())
        agreement = sum(
            int(max(range(len(lp)), key=lp.__getitem__) == final_arg)
            for lp in per_order_logps
        ) / len(per_order_logps)

        if isinstance(q, ChoiceQuestion):
            labels = canonical
        elif isinstance(q, NoulQuestion):
            labels = ["yes", "no"]
        else:
            labels = [str(i) for i in range(len(q.levels))]

        return build_answer(
            q,
            labels,
            pooled,
            sum(valids) / len(valids),
            agreement,
            self.calibrator,
        )

    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Prediction:
        qs = parse_questions(questions)
        q_map = {q.qid: q for q in qs}
        t0 = time.perf_counter()

        state_text = render_state(state)
        qtypes = {q.qtype for q in qs}
        prefix_prompts = {}
        saved_states = {}
        total_static_tokens = 0

        # Prefill state onto static KV cache for each question type
        for qt in qtypes:
            pb = self.prompts.get(qt, self.prompt) if getattr(self, "prompts", None) else self.prompt
            prefix_prompts[qt] = pb._static_text + state_text + pb._between
            total_static_tokens += len(self.tok.encode(pb._static_text, add_special_tokens=False))

            static_state = self._static_states.get(qt) or self._static_states.get("default")
            if static_state is not None:
                self.llm.load_state(static_state)
                state_tokens = self.tok.encode(state_text + pb._between, add_special_tokens=False)
                self.llm.eval(state_tokens)
                saved_states[qt] = self.llm.save_state()
            else:
                prefix_tokens = self.tok.encode(prefix_prompts[qt], add_special_tokens=False)
                self.llm.reset()
                self.llm.eval(prefix_tokens)
                saved_states[qt] = self.llm.save_state()

        t1 = time.perf_counter()

        answers = {
            q.qid: self._answer(q, prefix_prompts[q.qtype], saved_state=saved_states.get(q.qtype))
            for q in qs
        }
        t2 = time.perf_counter()

        state_tokens = len(self.tok.encode(state_text, add_special_tokens=False))

        return Prediction(
            answers=answers,
            timings_ms={"state": (t1 - t0) * 1e3, "questions": (t2 - t1) * 1e3, "total": (t2 - t0) * 1e3},
            tokens={"static_cached": total_static_tokens, "state": state_tokens},
            model=getattr(self, "model_name", "llama-cpp"),
            questions=q_map,
        )
