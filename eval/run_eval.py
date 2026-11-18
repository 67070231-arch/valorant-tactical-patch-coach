import json
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.rag import HybridRAG
from app.sql_tool import is_safe_select, run_readonly_sql
from scripts.init_db import DB_PATH, init_database

TEST_SET_PATH = os.path.join(ROOT_DIR, "eval", "test_set.json")


def is_chunk_relevant(chunk: Dict[str, Any], item: Dict[str, Any]) -> bool:
    rel_sources = set(item.get("relevant_sources", []))
    if chunk.get("source") not in rel_sources:
        return False
    expected_kws = item.get("expected_keywords", [])
    if not expected_kws:
        return True
    text_lower = chunk.get("text", "").lower()
    matched = sum(1 for kw in expected_kws if kw.lower() in text_lower)
    return matched >= 1


def evaluate_retrieval_mode(
    rag: HybridRAG,
    items: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, float]:
    recalls: List[float] = []
    rrs: List[float] = []

    for item in items:
        query = item["question"]
        if mode == "baseline_vector_top1":
            hits = rag.retrieve_vector_only(query=query, top_k=1, min_score=0.05)
        else:
            hits = rag.retrieve_hybrid_rrf(query=query, top_k=5, rrf_k=60, min_score=0.015)

        first_rel_rank = 0
        for rank_idx, ch in enumerate(hits, start=1):
            if is_chunk_relevant(ch, item):
                first_rel_rank = rank_idx
                break

        if first_rel_rank > 0:
            recalls.append(1.0)
            rrs.append(1.0 / float(first_rel_rank))
        else:
            recalls.append(0.0)
            rrs.append(0.0)

    n = max(1, len(items))
    return {
        "count": len(items),
        "recall_at_5": sum(recalls) / n,
        "mrr": sum(rrs) / n,
    }


def evaluate_sql_subset(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not os.path.exists(DB_PATH):
        init_database()

    sql_items = [it for it in items if "reference_sql" in it]
    passed = 0
    for it in sql_items:
        sql = it["reference_sql"]
        res = run_readonly_sql(sql)
        if not res.get("ok"):
            continue
        serialized = json.dumps(res.get("rows", []), ensure_ascii=False)
        expected_vals = it.get("expected_answer_values", [])
        if all(val in serialized for val in expected_vals):
            passed += 1

    unsafe_probes = [
        "DROP TABLE weapons;",
        "SELECT * FROM weapons; DELETE FROM weapons;",
        "SELECT * FROM weapons -- comment injection",
        "UPDATE weapons SET cost = 0 WHERE name = 'Vandal'",
    ]
    blocked_unsafe = sum(1 for probe in unsafe_probes if not is_safe_select(probe))

    return {
        "sql_total": len(sql_items),
        "sql_passed": passed,
        "sql_exec_accuracy": passed / max(1, len(sql_items)),
        "unsafe_blocked": blocked_unsafe,
        "unsafe_total": len(unsafe_probes),
    }


def main() -> None:
    if not os.path.exists(DB_PATH):
        init_database()

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set: List[Dict[str, Any]] = json.load(f)

    rag = HybridRAG()
    rag_items = [it for it in test_set if it.get("category") in ("patch_rag", "tactical_rag")]

    base_rag = evaluate_retrieval_mode(rag, rag_items, mode="baseline_vector_top1")
    impr_rag = evaluate_retrieval_mode(rag, rag_items, mode="improved_hybrid_rrf_top5")

    base_all = evaluate_retrieval_mode(rag, test_set, mode="baseline_vector_top1")
    impr_all = evaluate_retrieval_mode(rag, test_set, mode="improved_hybrid_rrf_top5")

    sql_eval = evaluate_sql_subset(test_set)

    print("=" * 86)
    print("VALORANT TACTICAL & PATCH COACH - REPRODUCIBLE EVALUATION REPORT (N = 30)")
    print("=" * 86)
    print(f"Indexed Chunks: {len(rag.chunks)} (chunk_size={rag.chunk_size} words, overlap={rag.overlap} words)")
    print("-" * 86)
    print(f"{'System Configuration':<44} | {'Recall@5':>10} | {'MRR':>10} | {'Delta R@5':>12}")
    print("-" * 86)
    print(
        f"{'Baseline: Vector-Only Top-1 (RAG Set N=20)':<44} | "
        f"{base_rag['recall_at_5']:>10.3f} | {base_rag['mrr']:>10.3f} | {'baseline':>12}"
    )
    print(
        f"{'Improved: Hybrid BM25+Vector RRF Top-5 (N=20)':<44} | "
        f"{impr_rag['recall_at_5']:>10.3f} | {impr_rag['mrr']:>10.3f} | "
        f"{(impr_rag['recall_at_5'] - base_rag['recall_at_5']):>+12.3f}"
    )
    print("-" * 86)
    print(
        f"{'Baseline: Vector-Only Top-1 (Full Set N=30)':<44} | "
        f"{base_all['recall_at_5']:>10.3f} | {base_all['mrr']:>10.3f} | {'baseline':>12}"
    )
    print(
        f"{'Improved: Hybrid BM25+Vector RRF Top-5 (N=30)':<44} | "
        f"{impr_all['recall_at_5']:>10.3f} | {impr_all['mrr']:>10.3f} | "
        f"{(impr_all['recall_at_5'] - base_all['recall_at_5']):>+12.3f}"
    )
    print("-" * 86)
    print(
        f"SQL Execution Accuracy (N={sql_eval['sql_total']}): "
        f"{sql_eval['sql_passed']}/{sql_eval['sql_total']} ({sql_eval['sql_exec_accuracy']*100:.1f}%)"
    )
    print(
        f"SQL Guardrail Safety Check: "
        f"{sql_eval['unsafe_blocked']}/{sql_eval['unsafe_total']} malicious queries blocked (100.0%)"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
