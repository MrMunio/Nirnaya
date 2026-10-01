"""Command line: `python -m nirnaya eval` and `python -m nirnaya demo`."""
from __future__ import annotations

import argparse
import json
import sys

from .calibration import Calibrator
from .engine import NirnayaEngine
from .evaluate import load_cases, print_report, run_eval
from .model_registry import registry, get_hardware_info


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="qwen3-4b", help="model alias (e.g. qwen3-4b, qwen3.5-4b-gguf, bonsai-27b) or HF model path")
    p.add_argument("--backend", default="auto", choices=["auto", "transformers", "torch", "prism", "prism_llama", "llama.cpp", "llama_cpp"], help="engine backend (auto, transformers, prism_llama, llama_cpp)")
    p.add_argument("--device", default="auto", help="execution device: auto (default), cuda, cuda:0, or cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--n-perms", type=int, default=None, help="option-order permutations for choice questions")
    p.add_argument("--ngl", "--n-gpu-layers", dest="n_gpu_layers", type=int, default=None, help="number of layers to offload to GPU in llama.cpp")
    p.add_argument("--port", type=int, default=None, help="server port for llama.cpp prism backend")
    p.add_argument("--no-few-shot", action="store_true")
    p.add_argument("--no-chat-template", action="store_true", help="use plain-text prompt (base models)")
    p.add_argument("--enable-thinking", action="store_true", help="enable thinking mode in reasoning models")
    p.add_argument("--threads", type=int, default=None, help="CPU threads")
    p.add_argument("--calibrator", default=None, help="path to a saved calibrator JSON")


def _engine(a) -> NirnayaEngine:
    kwargs = {
        "model_name": a.model,
        "backend": a.backend,
        "device": a.device,
        "dtype": a.dtype,
        "few_shot": not a.no_few_shot,
        "use_chat_template": False if a.no_chat_template else None,
        "enable_thinking": a.enable_thinking,
        "num_threads": a.threads,
        "calibrator": Calibrator.load(a.calibrator) if a.calibrator else None,
    }
    if getattr(a, "port", None) is not None:
        kwargs["port"] = a.port
    if a.n_perms is not None:
        kwargs["n_perms"] = a.n_perms
    if a.n_gpu_layers is not None:
        kwargs["n_gpu_layers"] = a.n_gpu_layers

    return NirnayaEngine(**kwargs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nirnaya")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="run the labeled test set")
    _add_model_args(ev)
    ev.add_argument("--data", default="data/test_cases.json")
    ev.add_argument("--calibrate", action="store_true", help="also report k-fold cross-fitted calibration")
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--out", default=None, help="write full JSON report here")

    dm = sub.add_parser("demo", help="one state, several typed questions")
    _add_model_args(dm)

    sub.add_parser("models", help="list all registered model configurations and hardware status")

    a = ap.parse_args(argv)

    if a.cmd == "models":
        hw = get_hardware_info()
        print("\n=== Nirnaya Platform Hardware ===")
        print(f"CUDA Available: {hw['cuda_available']}")
        print(f"Recommended Device: {hw['recommended_device']}")
        print(f"Default GPU Offload (NGL): {hw['recommended_ngl']}")
        print("\n=== Registered Models in models_config.json ===")
        for m in registry.list_models():
            print(f"  {m['alias']:<18} backend={m['backend']:<13} name={m['name']}")
            if m.get("description"):
                print(f"    |- {m['description']}")
        print("\nRun any model with:")
        print("  python -m nirnaya demo --model <alias>")
        print("  python -m nirnaya eval --model <alias> --data data/test_cases.json\n")
        return 0

    engine = _engine(a)

    if a.cmd == "demo":
        from examples.demo import DEMO_STATE, DEMO_QUESTIONS  # noqa: WPS433
        pred = engine.predict(DEMO_STATE, DEMO_QUESTIONS)
        for qid, ans in pred.answers.items():
            latency = pred.timings_ms.get(qid)
            lat_str = f" latency={latency:.1f}ms" if latency is not None else ""
            print(f"{qid:<12} value={ans.value!r:<18} conf={ans.confidence:.3f} "
                  f"valid_mass={ans.valid_mass:.3f} agree={ans.agreement:.2f}{lat_str}")
            print(f"{'':<12} " + ", ".join(f"{l}={p:.3f}" for l, p in zip(ans.labels, ans.probs)))
        print(f"\n{pred.timings_ms['total']:.0f} ms total  (state {pred.tokens['state']} tokens, state latency: {pred.timings_ms['state']:.1f}ms, all questions: {pred.timings_ms['questions']:.1f}ms)")
        return 0

    print(f"Running {a.data} ...")
    report = run_eval(engine, load_cases(a.data), calibrate=a.calibrate, folds=a.folds)
    print_report(report)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nfull report -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
