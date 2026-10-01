"""Client integration script to test a live Nirnaya Server (local or Docker)."""
import os
import sys
import time
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = os.getenv("NIRNAYA_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("NIRNAYA_API_KEY", "nir_live_root_secret_key_change_me")

print(f"Connecting to Nirnaya Server at: {BASE_URL}")

client = httpx.Client(timeout=90.0)

# 1. Health check
try:
    health_resp = client.get(f"{BASE_URL}/health")
    print(f"[1/4] Health Check: HTTP {health_resp.status_code} -> {health_resp.json()}")
except Exception as e:
    print(f"❌ Server connection failed: {e}")
    sys.exit(1)

# 2. Readiness check
ready_resp = client.get(f"{BASE_URL}/ready")
print(f"[2/4] Readiness Check: HTTP {ready_resp.status_code} -> {ready_resp.json()}")

# 3. List models
models_resp = client.get(
    f"{BASE_URL}/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print(f"[3/4] Models Endpoint: HTTP {models_resp.status_code}")
if models_resp.status_code == 200:
    for m in models_resp.json().get("models", []):
        default_tag = " (DEFAULT)" if m.get("is_default") else ""
        print(f"   • {m['alias']}{default_tag} [{m['backend']}]")

# 4. System-One Prediction Request
print("\n[4/4] Sending Decision Request to POST /v1/systemone ...")
payload = {
    "model": "qwen3-0.6b-gguf",
    "state": (
        "Server alert: Memory usage reached 98% on host prod-db-01. "
        "PostgreSQL connection pool exhausted. 45 queries timed out."
    ),
    "questions": {
        "urgency": {
            "type": "score",
            "instructions": "Rate how critical this infrastructure incident is from 0 (minor) to 4 (p0 disaster)",
            "criteria": ["p4 informational", "p3 low", "p2 medium", "p1 critical", "p0 outage disaster"]
        },
        "team": {
            "type": "choice",
            "instructions": "Which response team should be paged immediately?",
            "criteria": {
                "dba": "Database administrators and Postgres engineers",
                "frontend": "Frontend UI and client web engineers",
                "marketing": "Social media and marketing team"
            }
        },
        "page_oncall": {
            "type": "noul",
            "instructions": "Should an urgent on-call pager notification be sent right now?"
        }
    }
}

t0 = time.perf_counter()
resp = client.post(
    f"{BASE_URL}/v1/systemone",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json=payload,
)
elapsed_ms = (time.perf_counter() - t0) * 1000.0

if resp.status_code != 200:
    print(f"❌ Prediction request failed: HTTP {resp.status_code}\n{resp.text}")
    sys.exit(1)

data = resp.json()
print(f"✅ Prediction succeeded in {elapsed_ms:.1f}ms! (Model: {data.get('model')})")
print("\nDecision Results (TypeSafe Jev Response):")
for qid, ans in data.get("answers", {}).items():
    if ans["type"] == "choice":
        print(f"  • [{ans['type'].upper()}] {qid}: choice='{ans['choice']}' (confidence={ans['confidence']:.2f})")
    elif ans["type"] == "score":
        print(f"  • [{ans['type'].upper()}] {qid}: score={ans['score']:.2f} (confidence={ans['confidence']:.2f})")
    elif ans["type"] == "noul":
        print(f"  • [{ans['type'].upper()}] {qid}: P(true)={ans['noul']:.4f}")

print(f"\nToken Accounting: In={data['usage']['input_tokens']} | Out={data['usage']['output_tokens']}")
print("=" * 60)
print("🎉 ALL SYSTEMS FULLY OPERATIONAL!")
