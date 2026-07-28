"""Evaluation dashboard for the Agentic RAG Platform.

Reads eval/results/ir_scorecard.json (retrieval metrics) and, if present,
eval/results/answer_scores.jsonl (answer-quality LLM-judge scores).

Run:
    streamlit run eval/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path("eval/results")
IR_PATH = RESULTS_DIR / "ir_scorecard.json"
ANSWER_PATH = RESULTS_DIR / "answer_scores.jsonl"

C_DENSE = "#6b7280"
C_HYBRID = "#3b82f6"
C_RERANK = "#22c55e"

st.set_page_config(page_title="Agentic RAG - Eval Dashboard", page_icon="S", layout="wide")

st.title("Agentic RAG Platform - Evaluation Dashboard")
st.caption(
    "Retrieval quality, the rerank uplift, and answer faithfulness/correctness "
    "over a hand-labeled golden dataset of OWASP/NIST security questions."
)


def load_ir():
    if not IR_PATH.exists():
        return None
    return json.loads(IR_PATH.read_text(encoding="utf-8"))


ir = load_ir()

if ir is None:
    st.warning("No IR scorecard found. Run `python -m scripts.eval_retrieval` first.")
else:
    st.header("Retrieval quality by configuration")

    configs = list(ir.keys())
    metrics = ["mean_precision", "mean_recall", "mean_mrr", "mean_ndcg"]
    metric_labels = {"mean_precision": "P@5", "mean_recall": "R@5", "mean_mrr": "MRR", "mean_ndcg": "nDCG@5"}
    color_for = {}
    for c in configs:
        cl = c.lower()
        if "rerank" in cl:
            color_for[c] = C_RERANK
        elif "hybrid" in cl:
            color_for[c] = C_HYBRID
        else:
            color_for[c] = C_DENSE

    fig = go.Figure()
    for c in configs:
        fig.add_trace(go.Bar(
            name=c.strip(),
            x=[metric_labels[m] for m in metrics],
            y=[ir[c][m] for m in metrics],
            marker_color=color_for[c],
            text=[f"{ir[c][m]:.3f}" for m in metrics],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="score (higher is better)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420, margin=dict(t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    rerank_cfg = next((c for c in configs if "rerank" in c.lower()), None)
    hybrid_cfg = next((c for c in configs if "hybrid" in c.lower() and "rerank" not in c.lower()), None)

    if rerank_cfg and hybrid_cfg:
        st.subheader("Impact of the reranking stage")
        d_ndcg = ir[rerank_cfg]["mean_ndcg"] - ir[hybrid_cfg]["mean_ndcg"]
        d_mrr = ir[rerank_cfg]["mean_mrr"] - ir[hybrid_cfg]["mean_mrr"]
        d_prec = ir[rerank_cfg]["mean_precision"] - ir[hybrid_cfg]["mean_precision"]
        c1, c2, c3 = st.columns(3)
        c1.metric("nDCG@5", f"{ir[rerank_cfg]['mean_ndcg']:.3f}", f"{d_ndcg:+.3f}")
        c2.metric("MRR", f"{ir[rerank_cfg]['mean_mrr']:.3f}", f"{d_mrr:+.3f}")
        c3.metric("P@5", f"{ir[rerank_cfg]['mean_precision']:.3f}", f"{d_prec:+.3f}")
        st.caption(
            f"Deltas are hybrid (dense+BM25) -> hybrid+rerank. "
            f"n = {ir[rerank_cfg].get('n_questions', '?')} doc-answerable questions."
        )

    best_cfg = rerank_cfg or hybrid_cfg or configs[0]
    per_cat = ir[best_cfg].get("per_category", {})
    if per_cat:
        st.subheader(f"Per-category nDCG@5 - {best_cfg.strip()}")
        rows = []
        for cat, vals in per_cat.items():
            rows.append({
                "category": cat, "n": int(vals.get("n", 0)),
                "P@5": round(vals.get("precision", 0), 3),
                "R@5": round(vals.get("recall", 0), 3),
                "MRR": round(vals.get("mrr", 0), 3),
                "nDCG@5": round(vals.get("ndcg", 0), 3),
            })
        df = pd.DataFrame(rows).sort_values("nDCG@5", ascending=False)
        cat_fig = go.Figure(go.Bar(
            x=df["nDCG@5"], y=df["category"], orientation="h",
            marker_color=C_HYBRID,
            text=[f"{v:.2f}" for v in df["nDCG@5"]], textposition="outside",
        ))
        cat_fig.update_layout(
            xaxis=dict(range=[0, 1.05], title="nDCG@5"),
            height=max(220, 40 * len(df)), margin=dict(t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(cat_fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Latency by configuration"):
        for c in configs:
            secs = ir[c].get("elapsed_s")
            if secs is not None:
                st.write(f"**{c.strip()}** - {secs:.1f}s for {ir[c].get('n_questions', '?')} questions")
        st.caption(
            "Reranking is CPU-bound (cross-encoder scores every candidate pair). "
            "A GPU or a distilled reranker cuts this substantially."
        )


st.divider()
st.header("Answer quality (LLM-as-judge)")


def load_answers():
    if not ANSWER_PATH.exists():
        return None
    rows = [json.loads(l) for l in ANSWER_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows) if rows else None


ans = load_answers()

if ans is None or ans.empty:
    st.info(
        "No answer-quality scores yet. Run `python -m scripts.eval_answers` "
        "(needs LLM quota - best run via the Azure provider in one pass). "
        "Faithfulness = fraction of answer claims grounded in retrieved context; "
        "correctness = does the answer actually answer the question."
    )
else:
    n = len(ans)
    mean_f = ans["faithfulness"].mean()
    mean_c = ans["correctness"].mean()
    halluc = int((ans["faithfulness"] < 0.7).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean faithfulness", f"{mean_f:.3f}")
    c2.metric("Mean correctness", f"{mean_c:.3f}")
    c3.metric("Hallucination rate", f"{halluc / n:.0%}", help="faithfulness < 0.7")

    sc = go.Figure(go.Scatter(
        x=ans["faithfulness"], y=ans["correctness"],
        mode="markers+text",
        marker=dict(size=12, color=C_RERANK, line=dict(width=1, color="white")),
        text=ans["id"], textposition="top center",
        hovertext=ans.get("reasoning", ""),
    ))
    sc.update_layout(
        xaxis=dict(range=[-0.05, 1.05], title="faithfulness"),
        yaxis=dict(range=[-0.05, 1.05], title="correctness"),
        height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(sc, use_container_width=True)

    if "category" in ans.columns:
        st.subheader("Mean scores by category")
        cat_df = ans.groupby("category")[["faithfulness", "correctness"]].mean().round(3).reset_index()
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    flagged = ans[(ans["correctness"] < 0.5) | (ans["faithfulness"] < 0.7)]
    if not flagged.empty:
        st.subheader("Flagged answers (low faithfulness or correctness)")
        cols = [c for c in ["id", "category", "faithfulness", "correctness", "reasoning"] if c in flagged.columns]
        st.dataframe(flagged[cols], use_container_width=True, hide_index=True)


st.divider()
st.caption("Agentic RAG Platform - eval dashboard - data from eval/results/")