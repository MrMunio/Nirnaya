"""PrismLlamaEngine - Nirnaya backend for PrismML low-bit GGUF models via llama.cpp CUDA server.

Features:
- Runs natively on NVIDIA GPU with CUDA acceleration (e.g. RTX 4070 Laptop GPU).
- Supports Bonsai-27B (1-bit Q1_0_g128) and other PrismML GGUF architectures.
- Fully compatible with Nirnaya's prefix caching, permutation pooling, and calibration.
- Automatically manages local llama-server subprocess if not already running.
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.special

from .labels import LabelSpace, build_label_space
from .prompting import PromptBuilder, render_question, render_state
from .readout import build_answer, pool_logps
from .schema import Answer, ChoiceQuestion, NoulQuestion, Prediction, Question, parse_questions


class LlamaServerTokenizer:
    """Tokenizer client communicating with llama-server's /tokenize endpoint and jinja template."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._cache: Dict[str, List[int]] = {}
        # Fetch chat template from server /props
        req = urllib.request.Request(f"{base_url}/props")
        with urllib.request.urlopen(req) as resp:
            props = json.loads(resp.read().decode())
            self.chat_template = props.get("chat_template", "")
            self.bos_token = props.get("bos_token", "")
        import jinja2
        self._jinja_tmpl = jinja2.Environment().from_string(self.chat_template)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text in self._cache:
            return list(self._cache[text])
        req = urllib.request.Request(
            f"{self.base_url}/tokenize",
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            tokens = json.loads(resp.read().decode())["tokens"]
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


class PrismLlamaEngine:
    def __init__(
        self,
        model_name: str = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf",
        server_url: Optional[str] = None,
        port: int = 8088,
        n_gpu_layers: int = 99,
        n_ctx: int = 4096,
        n_perms: int = 3,
        few_shot: bool = True,
        use_chat_template: Optional[bool] = None,
        enable_thinking: bool = False,
        calibrator=None,
        auto_start: bool = True,
        dedicated_prompts: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        self.port = port
        self.base_url = server_url or f"http://127.0.0.1:{port}"
        self.n_perms = max(1, n_perms)
        self.calibrator = calibrator
        self.server_process: Optional[subprocess.Popen] = None
        self._spaces: Dict[tuple, LabelSpace] = {}
        self.dedicated_prompts = dedicated_prompts

        # Resolve model path
        resolved_path = self._resolve_model_path(model_name)
        self.model_path = resolved_path

        # Start server if needed
        if auto_start and not self._is_server_healthy():
            self._start_server(resolved_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)

        # Initialize tokenizer & prompt builder
        self.tok = LlamaServerTokenizer(self.base_url)
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

    def _resolve_model_path(self, model_name: str) -> str:
        # Check direct path
        if os.path.exists(model_name):
            return os.path.abspath(model_name)

        # Common fallback paths in project
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(proj_root, "models", "bonsai-27b", "Bonsai-27B-Q1_0.gguf")
        if os.path.exists(candidate):
            return candidate

        return model_name

    def _is_server_healthy(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                return data.get("status") == "ok"
        except Exception:
            return False

    def _start_server(self, model_path: str, n_gpu_layers: int, n_ctx: int):
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        bin_dir = os.path.join(proj_root, "bin", "prism_llama")
        server_exe = os.path.join(bin_dir, "llama-server.exe")

        if not os.path.exists(server_exe):
            raise FileNotFoundError(f"llama-server.exe not found at {server_exe}")

        # Set PATH to include torch/lib CUDA DLLs and prism bin dir
        env = os.environ.copy()
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        path_prefix = f"{bin_dir};{torch_lib};"
        env["PATH"] = path_prefix + env.get("PATH", "")

        cmd = [
            server_exe,
            "-m", model_path,
            "-ngl", str(n_gpu_layers),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-c", str(n_ctx),
            "--parallel", "1",
            "-sps", "0.1",
            "--reasoning-budget", "0",
        ]

        print(f"[PrismLlamaEngine] Starting server: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self.close)

        # Wait for health
        t0 = time.time()
        while time.time() - t0 < 180:
            if self._is_server_healthy():
                print(f"[PrismLlamaEngine] Server ready at {self.base_url}")
                return
            time.sleep(1.0)

        raise RuntimeError("llama-server failed to become healthy within 180 seconds.")

    def close(self):
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None

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

    def _answer(self, q: Question, prefix_input: Any) -> Answer:
        per_order_logps, valids = [], []
        canonical = list(q.options) if isinstance(q, ChoiceQuestion) else None
        pb = self.prompts.get(q.qtype, self.prompt) if getattr(self, "prompts", None) else self.prompt
        prefix_prompt = prefix_input[q.qtype] if isinstance(prefix_input, dict) else prefix_input

        for order in self._orders(q):
            qtext, label_strs = render_question(q, order)
            space = self._space(q.qtype, label_strs)

            suffix = qtext + pb._tail + pb.answer_cue + " "
            full_prompt = prefix_prompt + suffix

            body = {
                "prompt": full_prompt,
                "n_predict": 1,
                "n_probs": 300,
                "cache_prompt": True,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                f"{self.base_url}/completion",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode())

            top_probs = res["completion_probabilities"][0]["top_logprobs"]
            lp_dict = {t["id"]: t["logprob"] for t in top_probs}

            per_label_lp = []
            for token_ids in space.token_ids:
                found_lps = [lp_dict[tid] for tid in token_ids if tid in lp_dict]
                if found_lps:
                    per_label_lp.append(float(scipy.special.logsumexp(found_lps)))
                else:
                    per_label_lp.append(-30.0)

            valid_log_mass = float(scipy.special.logsumexp(per_label_lp))
            restricted = [lp - valid_log_mass for lp in per_label_lp]

            if canonical is not None:
                by_key = {key: restricted[i] for i, key in enumerate(order)}
                restricted = [by_key[key] for key in canonical]

            per_order_logps.append(restricted)
            valids.append(float(np.exp(valid_log_mass)))

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
        total_static_tokens = 0

        if getattr(self, "dedicated_prompts", False) and getattr(self, "prompts", None):
            for qt in qtypes:
                pb = self.prompts.get(qt, self.prompt)
                prefix_prompts[qt] = pb._static_text + state_text + pb._between
                total_static_tokens += len(self.tok.encode(pb._static_text, add_special_tokens=False))
        else:
            single_prefix = self.prompt._static_text + state_text + self.prompt._between
            for qt in qtypes:
                prefix_prompts[qt] = single_prefix
            total_static_tokens = len(self.tok.encode(self.prompt._static_text, add_special_tokens=False))

        t1 = time.perf_counter()

        answers = {}
        q_timings = {}
        for q in qs:
            t_q = time.perf_counter()
            answers[q.qid] = self._answer(q, prefix_prompts)
            q_timings[q.qid] = (time.perf_counter() - t_q) * 1e3
        t2 = time.perf_counter()

        state_tokens = len(self.tok.encode(state_text, add_special_tokens=False))

        timings = {
            "state": (t1 - t0) * 1e3,
            "questions": (t2 - t1) * 1e3,
            "total": (t2 - t0) * 1e3,
            **q_timings,
        }

        return Prediction(
            answers=answers,
            timings_ms=timings,
            tokens={"static_cached": total_static_tokens, "state": state_tokens},
            model=getattr(self, "model_name", "prism-bonsai-27b"),
            questions=q_map,
        )
