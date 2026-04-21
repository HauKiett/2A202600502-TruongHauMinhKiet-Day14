import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.runner import BenchmarkRunner

# ── Release gate thresholds (per spec in Role day 14.md) ─────────────────────
GATE_MIN_DELTA_AVG_SCORE    =  0.2   # V2 must improve avg_score by at least this
GATE_MAX_HIT_RATE3_DROP     =  0.02  # hit_rate@3 must not drop more than this
GATE_MIN_AGREEMENT_RATE     =  0.70  # V2 agreement_rate must meet this floor
GATE_MAX_P95_LATENCY_FACTOR =  1.2   # V2 p95 latency must stay within 1.2x V1


# ── Minimal passthrough evaluator ────────────────────────────────────────────

class PassthroughEvaluator:
    """Retrieval metrics are computed inside the runner; this is a no-op shim."""
    async def score(self, case: Dict, _resp: Dict) -> Dict:
        return {"faithfulness": 1.0, "relevancy": 1.0}


# ── Benchmark runner factory ──────────────────────────────────────────────────

def _make_runner(agent) -> BenchmarkRunner:
    judge = LLMJudge(
        model_a="gpt-4o-mini",
        model_b="gpt-4o",
        tiebreaker_model="gpt-4o",
    )
    return BenchmarkRunner(
        agent=agent,
        evaluator=PassthroughEvaluator(),
        judge=judge,
        semaphore_limit=10,
        timeout_s=30.0,
    )


# ── Load dataset ──────────────────────────────────────────────────────────────

def _load_dataset(path: str = "data/golden_set.jsonl") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Missing {path}. Run 'python data/synthetic_gen.py' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"[ERROR] {path} is empty.")
    return rows


# ── Single benchmark run ──────────────────────────────────────────────────────

async def run_benchmark(
    version: str,
    dataset: List[Dict],
    agent: Optional[MainAgent] = None,
) -> Tuple[List[Dict], Dict]:
    print(f"\n[>>] Running benchmark -- {version}  ({len(dataset)} cases)...")

    if agent is None:
        agent = MainAgent(version=version.split("_")[-1].lower())

    runner  = _make_runner(agent)
    t0      = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - t0

    stats = BenchmarkRunner.compute_summary_stats(results)

    print(
        f"  [OK] Done in {elapsed:.1f}s | "
        f"avg_score={stats.get('avg_score', 0):.2f} | "
        f"hit_rate@3={stats.get('hit_rate@3', 0):.2f} | "
        f"agreement_rate={stats.get('agreement_rate', 0):.2f} | "
        f"p95={stats.get('p95_latency_s', 0):.2f}s | "
        f"cost=${stats.get('total_cost_usd', 0):.4f}"
    )

    summary = {
        "metadata": {
            "version":    version,
            "total":      len(dataset),
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":  round(elapsed, 2),
        },
        "metrics": stats,
    }
    return results, summary


# ── Release gate ──────────────────────────────────────────────────────────────

def apply_release_gate(
    v1_metrics: Dict[str, Any],
    v2_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Release only when ALL conditions are satisfied:
      1. avg_score improved by >= GATE_MIN_DELTA_AVG_SCORE
      2. hit_rate@3 did not drop more than GATE_MAX_HIT_RATE3_DROP
      3. agreement_rate >= GATE_MIN_AGREEMENT_RATE
      4. p95_latency_s <= GATE_MAX_P95_LATENCY_FACTOR x V1
    """
    reasons: List[str] = []
    passed = True

    delta_score = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    if delta_score < GATE_MIN_DELTA_AVG_SCORE:
        reasons.append(
            f"avg_score delta {delta_score:+.4f} < required +{GATE_MIN_DELTA_AVG_SCORE} - FAIL"
        )
        passed = False
    else:
        reasons.append(f"avg_score delta {delta_score:+.4f} >= +{GATE_MIN_DELTA_AVG_SCORE} - OK")

    v1_hr3 = v1_metrics.get("hit_rate@3", v1_metrics.get("hit_rate", 0))
    v2_hr3 = v2_metrics.get("hit_rate@3", v2_metrics.get("hit_rate", 0))
    delta_hr3 = v2_hr3 - v1_hr3
    if delta_hr3 < -GATE_MAX_HIT_RATE3_DROP:
        reasons.append(
            f"hit_rate@3 dropped {abs(delta_hr3):.4f} > allowed {GATE_MAX_HIT_RATE3_DROP} - FAIL"
        )
        passed = False
    else:
        reasons.append(f"hit_rate@3 delta {delta_hr3:+.4f} within tolerance - OK")

    agr = v2_metrics.get("agreement_rate", 0)
    if agr < GATE_MIN_AGREEMENT_RATE:
        reasons.append(
            f"agreement_rate {agr:.4f} < required {GATE_MIN_AGREEMENT_RATE} - FAIL"
        )
        passed = False
    else:
        reasons.append(f"agreement_rate {agr:.4f} >= {GATE_MIN_AGREEMENT_RATE} - OK")

    v1_p95 = v1_metrics.get("p95_latency_s", 0)
    v2_p95 = v2_metrics.get("p95_latency_s", 0)
    latency_threshold = GATE_MAX_P95_LATENCY_FACTOR * v1_p95
    if v1_p95 > 0 and v2_p95 > latency_threshold:
        reasons.append(
            f"p95_latency {v2_p95:.3f}s > {GATE_MAX_P95_LATENCY_FACTOR}x V1 ({v1_p95:.3f}s) - FAIL"
        )
        passed = False
    else:
        reasons.append(f"p95_latency {v2_p95:.3f}s within {GATE_MAX_P95_LATENCY_FACTOR}x V1 - OK")

    return {
        "decision":          "APPROVE" if passed else "BLOCK",
        "passed":            passed,
        "delta_avg_score":   round(delta_score, 4),
        "delta_hit_rate3":   round(delta_hr3,   4),
        "v2_agreement_rate": round(agr,         4),
        "v2_p95_latency_s":  round(v2_p95,      4),
        "reasons":           reasons,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Load dataset once (shared for V1 and V2)
    try:
        dataset = _load_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return

    # ── V1 baseline ───────────────────────────────────────────────────────────
    _, v1_summary = await run_benchmark(
        "Agent_V1_Base", dataset, agent=MainAgent(version="v1")
    )

    # ── V2 optimized ──────────────────────────────────────────────────────────
    v2_results, v2_summary = await run_benchmark(
        "Agent_V2_Optimized", dataset, agent=MainAgent(version="v2")
    )

    # ── Cohen's Kappa across all V2 cases ────────────────────────────────────
    valid_v2 = [r for r in v2_results if r.get("status") != "error"]
    if valid_v2:
        model_keys = list(valid_v2[0]["judge"].get("individual_scores", {}).keys())
        if len(model_keys) >= 2:
            scores_a = [r["judge"]["individual_scores"].get(model_keys[0], 3) for r in valid_v2]
            scores_b = [r["judge"]["individual_scores"].get(model_keys[1], 3) for r in valid_v2]
            kappa = LLMJudge.cohen_kappa(scores_a, scores_b)
            v2_summary["metrics"]["cohen_kappa"] = kappa

    # ── Failure clustering ────────────────────────────────────────────────────
    clusters = BenchmarkRunner.cluster_failures(v2_results)

    # ── Release gate ──────────────────────────────────────────────────────────
    gate = apply_release_gate(v1_summary["metrics"], v2_summary["metrics"])

    print("\n[STATS] -- REGRESSION COMPARISON --")
    print(f"  V1 avg_score      : {v1_summary['metrics']['avg_score']:.4f}")
    print(f"  V2 avg_score      : {v2_summary['metrics']['avg_score']:.4f}")
    print(f"  delta             : {gate['delta_avg_score']:+.4f}")
    print(f"  V2 hit_rate@3     : {v2_summary['metrics'].get('hit_rate@3', 0):.4f}")
    print(f"  V2 agreement_rate : {gate['v2_agreement_rate']:.4f}")
    kappa_val = v2_summary["metrics"].get("cohen_kappa", "N/A")
    print(f"  V2 cohen_kappa    : {kappa_val}")
    print(f"  V2 p95_latency    : {gate['v2_p95_latency_s']:.3f}s")
    print(f"  V2 total_cost     : ${v2_summary['metrics'].get('total_cost_usd', 0):.4f}")

    verdict = "[APPROVE]" if gate["passed"] else "[BLOCK]"
    print(f"\n{verdict}  RELEASE GATE: {gate['decision']}")
    for reason in gate["reasons"]:
        prefix = "  [OK]" if "OK" in reason else "  [FAIL]"
        print(f"{prefix} {reason}")

    print("\n[CLUSTERS] Failure clusters (V2):")
    for cluster, case_ids in clusters.items():
        if case_ids:
            print(f"  {cluster}: {case_ids}")

    # ── Save reports ──────────────────────────────────────────────────────────
    v2_summary["regression"] = {
        "v1_metrics":   v1_summary["metrics"],
        "v2_metrics":   v2_summary["metrics"],
        "release_gate": gate,
        "failure_clusters": clusters,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    print("\n[SAVED] reports/summary.json, reports/benchmark_results.json")
    print("[NEXT]  Run 'python check_lab.py' to verify submission format.")


if __name__ == "__main__":
    asyncio.run(main())
