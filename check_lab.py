import json
import os

def validate_lab():
    print("[CHECK] Dang kiem tra dinh dang bai nop...")

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "analysis/failure_analysis.md"
    ]

    # 1. Kiểm tra sự tồn tại của tất cả file
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"[OK]  Tim thay: {f}")
        else:
            print(f"[MISS] Thieu file: {f}")
            missing.append(f)

    if missing:
        print(f"\n[FAIL] Thieu {len(missing)} file. Hay bo sung truoc khi nop bai.")
        return

    # 2. Kiểm tra nội dung summary.json
    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[FAIL] File reports/summary.json khong phai JSON hop le: {e}")
        return

    if "metrics" not in data or "metadata" not in data:
        print("[FAIL] File summary.json thieu truong 'metrics' hoac 'metadata'.")
        return

    metrics = data["metrics"]

    print(f"\n--- Summary stats ---")
    print(f"Total cases : {data['metadata'].get('total', 'N/A')}")
    print(f"Avg score   : {metrics.get('avg_score', 0):.2f}")

    # EXPERT CHECKS
    has_retrieval = "hit_rate" in metrics
    if has_retrieval:
        print(f"[OK]  Retrieval Metrics found (Hit Rate: {metrics['hit_rate']*100:.1f}%)")
    else:
        print(f"[WARN] Thieu Retrieval Metrics (hit_rate).")

    has_multi_judge = "agreement_rate" in metrics
    if has_multi_judge:
        print(f"[OK]  Multi-Judge Metrics found (Agreement Rate: {metrics['agreement_rate']*100:.1f}%)")
    else:
        print(f"[WARN] Thieu Multi-Judge Metrics (agreement_rate).")

    if data["metadata"].get("version"):
        print(f"[OK]  Agent version info found (Regression Mode)")

    print("\n[READY] Bai lab da san sang de cham diem!")

if __name__ == "__main__":
    validate_lab()
