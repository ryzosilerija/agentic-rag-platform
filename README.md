# Agentic RAG Platform

**A production-grade, multi-agent AI system for cybersecurity intelligence.**

Ask a security question in plain English. A supervisor routes it to the right specialist agent — one that searches security documentation, one that safely queries a vulnerability database, or one that calls live external APIs. For questions that need more than one source, a planner decomposes the task and coordinates several agents. Every answer is grounded, cited, traced, cost-accounted, and continuously evaluated.

Built over **~2,900 document chunks spanning four security taxonomies** (OWASP, NIST, MITRE CWE, CAPEC, and ATT&CK), the platform demonstrates the full lifecycle of modern LLM engineering: retrieval at scale, multi-agent orchestration, tool use, safety, observability, evaluation, cost-awareness, and cloud deployment.

<p align="center">
  <em>Python · LangGraph · Qdrant · FastAPI · OpenTelemetry · Model Context Protocol · Azure OpenAI · Next.js · Terraform</em>
</p>

<!-- Add badges here once CI is public, e.g.:
![eval](https://github.com/ryzosilerija/agentic-rag-platform/actions/workflows/eval.yml/badge.svg)
-->

---

## Table of Contents

- [Demo](#demo)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [The retrieval-at-scale study](#retrieval-at-scale)
- [The three agents](#the-three-agents)
- [Agent-to-agent orchestration](#a2a)
- [Safety: defense-in-depth, verified by red-teaming](#safety)
- [Observability, cost, and evaluation](#observability)
- [Results at a glance](#results)
- [Tech stack](#tech-stack)
- [Repository layout](#layout)
- [Quickstart](#quickstart)
- [Deployment](#deployment)
- [Honest limitations](#limitations)
- [Roadmap](#roadmap)

---

<a name="demo"></a>
## Demo

The **Sentinel** console shows live agent routing — every answer is tagged with the specialist that produced it (blue = RAG, green = SQL, amber = API), with an evidence trail of citations.

<!-- SCREENSHOTS: add three here, one per route color -->
<!--
![RAG route](docs/screenshots/route-rag.png)
![SQL route](docs/screenshots/route-sql.png)
![API route](docs/screenshots/route-api.png)
-->

> _Example:_ "Which vendor has the most known exploited vulnerabilities, and what are they known for?" → the planner queries the vulnerability database (finds **Microsoft, 382 entries**), feeds that into the docs agent to explain it, and composes a single grounded answer.

---

<a name="highlights"></a>
## Highlights

- **Multi-agent, not single-agent.** A LangGraph supervisor routes between three specialist agents (retrieval, text-to-SQL, live REST) at **100% routing accuracy** on a labeled eval set; an A2A planner coordinates several agents for multi-source questions.
- **A measured retrieval-at-scale study.** As the corpus grew **4.4×** across four security taxonomies, cross-encoder reranking's contribution to nDCG@5 **grew from +0.04 to +0.10** — a controlled result showing reranking matters *more* at scale, not less.
- **Safety is architectural, not prompt-based.** Text-to-SQL uses a read-only connection + SQL parser + schema allowlist; the REST agent has DNS-time SSRF protection. A **12-attack red-team suite is defended 100%**.
- **Everything is measured.** Five evaluation types (IR metrics, LLM-as-judge answer quality, routing accuracy, adversarial defense, reranker A/B) run **CI-gated on every pull request**.
- **Production concerns, not just a demo.** OpenTelemetry tracing, per-query cost/latency/cache metrics behind a `/metrics` endpoint, a provider-agnostic LLM abstraction (Azure / Gemini / local), Terraform infrastructure-as-code, and a clickable Next.js frontend.

---

<a name="architecture"></a>
## Architecture

```
                          ┌────────────────────────────┐
                          │   Next.js console (Sentinel)│
                          │  live agent-routing badges  │
                          └──────────────┬─────────────┘
                                         │  POST /chat
                          ┌──────────────▼─────────────┐
                          │      FastAPI backend        │
                          │  /chat · /chat/stream · /metrics
                          └──────────────┬─────────────┘
                                         │
                     ┌───────────────────▼────────────────────┐
                     │       A2A PLANNER-ORCHESTRATOR          │
                     │  decompose → dispatch → compose         │
                     │  (multi-source questions only)          │
                     └───────────────────┬────────────────────┘
                                         │
                          ┌──────────────▼─────────────┐
                          │        SUPERVISOR           │
                          │   LLM router: rag/sql/api   │
                          └──┬───────────┬───────────┬──┘
                             │           │           │
                ┌────────────▼──┐  ┌─────▼──────┐  ┌─▼───────────┐
                │  RAG AGENT    │  │ SQL AGENT  │  │  API AGENT  │
                │ hybrid + rerank│  │ text-to-SQL│  │ guarded REST│
                │ + MCP tools   │  │ read-only  │  │ SSRF-safe   │
                └───────┬───────┘  └─────┬──────┘  └──────┬──────┘
                        │                │                │
              ┌─────────▼──────┐  ┌──────▼───────┐  ┌─────▼──────┐
              │  Qdrant (2,900 │  │ SQLite (KEV  │  │ public     │
              │  chunks, 4     │  │ catalog,     │  │ REST APIs  │
              │  taxonomies)   │  │ ~1,400 rows) │  │ (NVD,GitHub)│
              └────────────────┘  └──────────────┘  └────────────┘

   Cross-cutting: OpenTelemetry tracing → Jaeger · cost/latency/cache metrics
                  · CI-gated evaluation harness · provider-agnostic LLM factory
```

The unifying design principle: **every agent implements the same `Agent` interface**, so the orchestration layers treat them uniformly and adding a fourth agent is *register + one router-prompt line*. That's why the system grew from one agent to an orchestrated three-agent platform without rewriting the earlier parts.

---

<a name="retrieval-at-scale"></a>
## The retrieval-at-scale study

Most RAG projects report a single retrieval number on a small corpus. This one treats retrieval as an experiment: the corpus was grown across four sizes, and the contribution of each retrieval component was measured at each size.

| Corpus size | Sources added | Hybrid nDCG@5 | +Rerank nDCG@5 | **Rerank Δ nDCG** | Rerank Δ MRR |
|---|---|---|---|---|---|
| 664 | OWASP + NIST | ~0.83 | 0.906 | **+0.040** | +0.072 |
| 1,633 | + MITRE CWE | 0.780 | 0.873 | **+0.093** | +0.104 |
| 2,945 | + CAPEC + ATT&CK | 0.754 | 0.851 | **+0.098** | +0.118 |

**The finding:** as the corpus grew 4.4×, cheap hybrid retrieval *degraded* (more distractors entered the candidate pool) — but the cross-encoder reranker's contribution **grew monotonically**. 

> **Reranking matters more at scale, not less.** The larger and noisier your candidate pool, the more a precise re-ranking pass earns its computational cost.

**An honest secondary finding:** BM25 fusion's contribution went slightly *negative* on the larger, more homogeneous corpus — it added distractor noise that the reranker then had to correct. This is a real characteristic of hybrid retrieval on dense-domain corpora, surfaced rather than hidden.

Retrieval architecture: **dense vectors (Qdrant, local BGE embeddings) + BM25 → Reciprocal Rank Fusion → cross-encoder reranking (`bge-reranker-base`)**. Fast-and-rough candidate generation, then a slow-and-precise quality pass on the top candidates.

---

<a name="the-three-agents"></a>
## The three agents

### RAG agent — grounded answers from documents
A LangGraph pipeline: **rewrite** the query → **retrieve** (hybrid) → **rerank** (cross-encoder) → **decide tools** → **synthesize** with inline `[N]` citations. It answers conceptual and how-to questions from the security corpus, and can call **MCP tools** for live facts (a specific CVE's severity from NVD, or exploitation status from the CISA KEV catalog) — firing conservatively only when the question needs current data.

### SQL agent — safe quantitative answers
Translates natural-language questions ("How many Microsoft vulnerabilities are in the catalog?") into SQL over the CISA Known Exploited Vulnerabilities catalog (~1,400 rows). Safety is **defense-in-depth** (see below). Cross-validated: the SQL agent and the MCP KEV tool independently agree on counts (Microsoft = 382).

### API agent — safe live external data
A *general* REST agent: it plans an HTTP request from the question, executes it under strict guards, and summarizes the result. It can hit any public API but **cannot reach internal or metadata endpoints** (see SSRF defense below).

---

<a name="a2a"></a>
## Agent-to-agent orchestration

Routing picks *one* agent. Some questions need *several* — e.g. "Which vendor has the most exploited vulnerabilities, and what are they known for?" is a database question (find the top vendor) **followed by** a docs question (explain it).

The **A2A protocol** defines typed message envelopes (`A2ATask`, `A2AResult` with an ok/error/refused status). The **planner-orchestrator**:
1. **Plans** — an LLM decomposes the question into an ordered list of sub-tasks, each assigned to an agent (simple questions → one task; complex → several, possibly dependent).
2. **Dispatches** — each sub-task is sent as an `A2ATask`; dependent tasks run after their dependencies, with the dependency's answer **injected into the dependent task's query**.
3. **Composes** — an LLM merges the results into one coherent answer.

This is genuine agent-to-agent cooperation: **one agent's output becomes another's input**, mediated by a typed protocol.

---

<a name="safety"></a>
## Safety: defense-in-depth, verified by red-teaming

### Text-to-SQL — three independent layers
Any one of these alone would block a malicious query:
1. **Read-only connection** — the database is opened in SQLite's read-only mode at the OS level. No write can happen even if everything else failed.
2. **SQL parsing** — generated SQL is parsed with `sqlglot` and rejected unless it's a *single SELECT statement*. No stacked queries, no DML/DDL, no PRAGMA/ATTACH.
3. **Schema allowlist** — every referenced table must be on an allowlist.

Because safety is **architectural, not prompt-based**, wrapping an attack in polite natural language ("...then delete all rows") doesn't defeat it — the guard validates the *generated SQL* regardless of phrasing.

### SSRF protection for the REST agent
A general "fetch any URL" agent is an SSRF liability. The guard applies: scheme allowlist (http/https), method allowlist (GET/HEAD), port allowlist, no redirect-following, and — critically — **DNS-resolution-time rejection of any private, loopback, link-local, or reserved IP**, blocking `127.0.0.1`, `10.x`, `192.168.x`, and the cloud-metadata endpoint `169.254.169.254` (the classic credential-theft target).

### Red-team evaluation — 12 attacks, 100% defended
A unified suite attacks all three entry points: NL-wrapped SQL injection, schema exfiltration, UNION-to-secrets, prompt injection, system-prompt exfiltration, jailbreaks (DAN), instruction override, and routing manipulation. Scoring uses **concrete checkable signals** (did DML/DDL reach the generated SQL? did the output contain an injection marker or exploit payload?) rather than a fuzzy LLM judge.

---

<a name="observability"></a>
## Observability, cost, and evaluation

**Distributed tracing.** Every pipeline stage emits an OpenTelemetry span (latency, token counts, tool arguments) nested into a per-request trace tree, exported to **Jaeger**. Optional LangSmith integration for LLM-native tracing. FastAPI is auto-instrumented.

**Cost & performance.** A cost layer converts per-call token counts into estimated cost-per-query, tracks cache-hit rate and latency, and exposes a live **`/metrics`** endpoint. It surfaced, for example, that cold-start model loading dominates first-request latency — a concrete "pre-warm models in production" insight.

**CI-gated evaluation harness** — five evaluation types, run on every push/PR via GitHub Actions:
- **Information-retrieval metrics** — Precision@k, Recall@k, MRR, nDCG@k on a hand-labeled golden dataset.
- **Answer quality** — an LLM-as-judge scores faithfulness and correctness.
- **Routing accuracy** — labeled question→agent pairs with a confusion matrix.
- **Adversarial defense** — the red-team suite above.
- **Reranker A/B** — base vs LoRA-fine-tuned, controlled before/after.

---

<a name="results"></a>
## Results at a glance

**Retrieval (production config, full 2,945-chunk corpus):**
| Metric | Value |
|---|---|
| Precision@5 | 0.666 |
| Recall@5 | 0.891 |
| MRR | 0.884 |
| nDCG@5 | 0.851 |

**Routing accuracy:** 16/16 = **100%** on the labeled set.
**Adversarial defense:** 12/12 = **100%** across SQL, RAG, and supervisor surfaces.
**Reranking value:** grows from +0.04 to +0.10 nDCG@5 as the corpus scales 4.4×.

---

<a name="tech-stack"></a>
## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (state-graph agents), custom supervisor + A2A planner |
| Retrieval | Qdrant (vectors), rank-bm25, Reciprocal Rank Fusion, BGE cross-encoder reranker |
| Embeddings | BGE `bge-base-en-v1.5` (local, no per-token cost) |
| LLM providers | Azure OpenAI (gpt-5-mini), Google Gemini, local — via a provider factory |
| Tools | Custom MCP server (NVD CVE lookup, CISA KEV search) |
| API | FastAPI (`/chat`, `/chat/stream` SSE, `/metrics`) |
| Frontend | Next.js 14 |
| Observability | OpenTelemetry → Jaeger; optional LangSmith |
| Data safety | sqlglot (SQL parsing), custom SSRF guard |
| Fine-tuning | PEFT / LoRA (reranker adapter) |
| Deployment | Docker (multi-stage, non-root), Terraform (Azure Container Apps) |
| CI | GitHub Actions (evaluation gate) |

---

<a name="layout"></a>
## Repository layout

```
src/
  agents/          rag_agent, sql_agent, api_agent, base (shared interface)
  orchestrator/    supervisor (routing), planner + a2a (orchestration)
  retrieval/       dense, bm25, fusion, rerank, hybrid
  ingestion/       loaders, chunking, index + corpus loaders (cwe/capec/attack)
  db/              database (KEV → SQLite), sql_guard (defense-in-depth)
  tools/           http_guard (SSRF), http_exec
  mcp_server/      MCP server + CVE/KEV tools
  llm/             provider factory, embeddings
  observability/   tracing, spans, cost
  api/             FastAPI app
eval/
  golden/          hand-labeled datasets (retrieval, routing)
  adversarial/     red-team attack suite
  rerank/          LoRA training pairs
  metrics_ir, metrics_answer, schemas, dashboard
scripts/           ingest_*, eval_*, build_db, train_reranker_lora, ...
frontend/          Next.js console (Sentinel)
infra/             Terraform (Azure Container Apps)
Dockerfile         multi-stage backend image
```

---

<a name="quickstart"></a>
## Quickstart

**Prerequisites:** Python 3.12, Docker, Node 18+ (for the frontend).

```bash
# 1. Start supporting services (Qdrant + Jaeger)
docker compose up -d

# 2. Install
pip install -e .

# 3. Configure — copy .env.example to .env and set a provider
#    PROVIDER=gemini (free) or PROVIDER=azure, plus the matching keys

# 4. Ingest the corpus
python -m scripts.ingest          # OWASP + NIST base corpus
python -m scripts.ingest_cwe      # + MITRE CWE
python -m scripts.ingest_capec    # + CAPEC
python -m scripts.ingest_attack   # + MITRE ATT&CK
python -m scripts.build_db        # KEV catalog → SQLite

# 5. Run the backend
uvicorn src.api.main:app --port 8000

# 6. (optional) Run the frontend
cd frontend && npm install && npm run dev   # → http://localhost:3000
```

**Try it:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I prevent SQL injection?"}'
```

**Evaluate:**
```bash
python -m scripts.eval_retrieval     # IR metrics (dense vs hybrid vs rerank)
python -m scripts.eval_routing       # routing accuracy + confusion matrix
python -m scripts.eval_adversarial   # red-team suite
```

**Observe:** open Jaeger at `http://localhost:16686`, or the cost snapshot at `http://localhost:8000/metrics`.

---

<a name="deployment"></a>
## Deployment

Infrastructure-as-code for Azure Container Apps lives in `infra/`:

```bash
cd infra
terraform init
terraform validate
cp terraform.tfvars.example terraform.tfvars   # fill in secrets
terraform plan
terraform apply
terraform output backend_url
```

Provisions a resource group, container registry, log analytics, a Container Apps environment, and the backend app (scale-to-zero, HTTPS ingress, LLM keys injected as secrets — never baked into the image). The backend image is built from the multi-stage, non-root `Dockerfile`.

---

<a name="limitations"></a>
## Honest limitations

A system is only as trustworthy as its author's honesty about its edges:

- **Answer-quality faithfulness measurement is noisy.** The LLM-judge model spent reasoning tokens and truncated some of its own scoring output (a correct Log4Shell answer scored 0 purely from truncation). The harness surfaced *both* a real faithfulness gap in multi-source categories *and* a measurement artifact — distinguishing the two is the skill. Correctness (~0.73) is the more reliable number.
- **The LoRA reranker fine-tune regressed retrieval.** Against a near-ceiling base model with a small domain dataset, the adapter overfit and slightly degraded nDCG. The evaluation harness caught it, and the base reranker was kept in production. Fine-tuning is not free — and disciplined evaluation catches regressions before they ship.
- **BM25 fusion adds distractor noise on the larger, homogeneous corpus** — its contribution goes slightly negative at scale, corrected by the reranker (see the scaling study).

None of these are failures. They're the findings a real evaluation process produces — and surfacing, explaining, and acting on them is what distinguishes engineering from demo-building.

---

<a name="roadmap"></a>
## Roadmap

Possible extensions (deliberately not built, to keep the project focused and honest about scope):
- Scale the corpus to 10k+ chunks with CVE feeds (volume; the scaling story is already established at 4 sizes).
- Streaming execution of planner → agent → partial synthesis.
- Conversation and long-term memory.
- Human-in-the-loop approval for write-capable agents.
- Load testing and Kubernetes deployment.

---

*Built as a portfolio flagship demonstrating end-to-end LLM engineering. Every metric in this README is measured and reproducible via the scripts above.*
