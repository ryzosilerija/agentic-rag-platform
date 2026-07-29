"use client";

import { useEffect, useRef, useState } from "react";

type Citation = { source_id: string; section: string; text: string };
type Msg = {
  role: "user" | "assistant";
  content: string;
  route?: string;
  citations?: Citation[];
};

const EXAMPLES = [
  { q: "How do I prevent SQL injection?", hint: "-> rag" },
  { q: "How many Microsoft vulnerabilities are in the KEV catalog?", hint: "-> sql" },
  { q: "How many stars does the langchain-ai/langchain GitHub repo have?", hint: "-> api" },
];

function routeClass(route?: string) {
  if (route === "sql") return "route-sql";
  if (route === "api") return "route-api";
  return "route-rag";
}
function routeLabel(route?: string) {
  if (route === "sql") return "SQL Agent";
  if (route === "api") return "API Agent";
  return "RAG Agent";
}

export default function Console() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer || "(no answer)",
          route: data.metadata?.routed_to,
          citations: data.citations || [],
        },
      ]);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "Couldn''t reach the platform. Make sure the backend is running:\n  uvicorn src.api.main:app --port 8000",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <header className="header">
        <div className="brand">
          sentinel<span className="dot">.</span>
          <span className="sub">agentic security console</span>
        </div>
        <div className="agents-legend">
          <div className="legend-item">
            <span className="legend-swatch" style={{ background: "var(--rag)" }} />
            RAG
          </div>
          <div className="legend-item">
            <span className="legend-swatch" style={{ background: "var(--sql)" }} />
            SQL
          </div>
          <div className="legend-item">
            <span className="legend-swatch" style={{ background: "var(--api)" }} />
            API
          </div>
        </div>
      </header>

      <div className="chat" ref={chatRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Ask a security question - the supervisor routes it live</h2>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex.q} className="example" onClick={() => send(ex.q)}>
                  {ex.q} <span className="tag">{ex.hint}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.role === "assistant" && m.route && (
              <div className={`route-badge ${routeClass(m.route)}`}>
                <span className="pulse" />
                {routeLabel(m.route)}
              </div>
            )}
            <div className="bubble">{m.content}</div>
            {m.citations && m.citations.length > 0 && (
              <div className="evidence">
                <div className="evidence-label">
                  evidence · {m.citations.length} source{m.citations.length > 1 ? "s" : ""}
                </div>
                {m.citations.slice(0, 5).map((c, j) => (
                  <div key={j} className="citation">
                    <span className="src">{c.source_id}</span>
                    {c.section ? ` · ${c.section}` : ""}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="msg assistant">
            <div className="thinking">
              <span className="spinner" />
              routing &amp; running agent…
            </div>
          </div>
        )}
      </div>

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(input)}
          placeholder="Ask about a vulnerability, a CVE count, or live repo data…"
          disabled={loading}
        />
        <button onClick={() => send(input)} disabled={loading || !input.trim()}>
          SEND
        </button>
      </div>
    </div>
  );
}