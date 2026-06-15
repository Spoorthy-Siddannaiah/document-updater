"use client";

import { useMemo, useState } from "react";
import {
  postQuery,
  setStatus,
  editSuggestion,
  regenerateSuggestion,
  saveSession,
  Suggestion,
} from "@/lib/api";
import SuggestionCard from "@/components/SuggestionCard";

const EXAMPLE =
  "WebSearchTool now requires a mandatory api_key argument when constructed";

export default function Home() {
  const [query, setQuery] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [considered, setConsidered] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [hasRun, setHasRun] = useState(false);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const groups = useMemo(() => {
    const m = new Map<string, Suggestion[]>();
    for (const s of suggestions) {
      const arr = m.get(s.file) ?? [];
      arr.push(s);
      m.set(s.file, arr);
    }
    return Array.from(m.entries());
  }, [suggestions]);

  function toggleFile(file: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      next.has(file) ? next.delete(file) : next.add(file);
      return next;
    });
  }

  const stats = useMemo(() => {
    const by = (p: (s: Suggestion) => boolean) => suggestions.filter(p).length;
    return {
      total: suggestions.length,
      approved: by((s) => s.status === "approved"),
      rejected: by((s) => s.status === "rejected"),
      saved: by((s) => s.status === "saved"),
      pending: by((s) => s.status === "pending"),
      high: by((s) => s.confidence === "high"),
      medium: by((s) => s.confidence === "medium"),
      low: by((s) => s.confidence === "low"),
    };
  }, [suggestions]);
  const approvedCount = stats.approved;

  function patch(updated: Suggestion) {
    setSuggestions((prev) =>
      prev.map((s) => (s.id === updated.id ? updated : s))
    );
  }

  async function runQuery() {
    setLoading(true);
    setError(null);
    setToast(null);
    try {
      const res = await postQuery(query);
      setSessionId(res.session_id);
      setSuggestions(res.suggestions);
      setConsidered(res.considered_chunks);
      setElapsedMs(res.elapsed_ms);
      setHasRun(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleStatus(id: string, action: "approve" | "reject") {
    if (!sessionId) return;
    try {
      patch(await setStatus(sessionId, id, action));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update");
    }
  }

  async function handleEdit(id: string, text: string) {
    if (!sessionId) return;
    patch(await editSuggestion(sessionId, id, text));
  }

  async function handleRegenerate(id: string) {
    if (!sessionId) return;
    try {
      patch(await regenerateSuggestion(sessionId, id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to regenerate");
    }
  }

  async function bulk(
    action: "approve" | "reject",
    pick: (s: Suggestion) => boolean
  ) {
    const targets = suggestions.filter(
      (s) => s.status !== "saved" && pick(s)
    );
    for (const s of targets) await handleStatus(s.id, action);
  }

  async function handleSave() {
    if (!sessionId) return;
    setSaving(true);
    setError(null);
    try {
      const res = await saveSession(sessionId);
      setSuggestions((prev) =>
        prev.map((s) => {
          const hit = res.saved.find((x) => x.id === s.id);
          return hit ? hit : s;
        })
      );
      setToast(
        res.saved.length
          ? `Saved ${res.saved.length} edit(s) across ${res.files.length} file(s): ${res.files.join(", ")}`
          : "No approved edits to save."
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="container">
      <h1>Pluno Doc Updater</h1>
      <p className="subtitle">
        Describe a product change. The AI finds documentation sections that need
        updating and proposes edits for your review.
      </p>

      <div className="query-form">
        <textarea
          rows={3}
          placeholder={EXAMPLE}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="bar">
          <button
            className="primary"
            onClick={runQuery}
            disabled={loading || !query.trim()}
          >
            {loading ? "Analyzing…" : "Suggest updates"}
          </button>
          <button onClick={() => setQuery(EXAMPLE)} disabled={loading}>
            Use example
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}
      {toast && <div className="toast">{toast}</div>}

      {hasRun && (
        <>
          <div className="statsbar">
            <span className="stat">{stats.total} suggestions</span>
            <span className="stat pend">{stats.pending} pending</span>
            <span className="stat appr">{stats.approved} approved</span>
            <span className="stat rej">{stats.rejected} rejected</span>
            {stats.saved > 0 && (
              <span className="stat saved">{stats.saved} saved</span>
            )}
            <span className="sep">·</span>
            <span className="stat">
              confidence: <b className="high">{stats.high}</b> high{" "}
              <b className="medium">{stats.medium}</b> medium{" "}
              <b className="low">{stats.low}</b> low
            </span>
            <span className="muted">· {considered} sections considered</span>
            {elapsedMs > 0 && (
              <span className="muted">· {(elapsedMs / 1000).toFixed(1)}s</span>
            )}
          </div>

          {suggestions.length > 0 && (
            <div className="bar actionbar">
              <button
                onClick={() => bulk("approve", (s) => s.confidence === "high")}
                disabled={stats.high === 0}
              >
                ✓ Approve all high ({stats.high})
              </button>
              <button
                onClick={() => bulk("reject", (s) => s.confidence === "low")}
                disabled={stats.low === 0}
              >
                ✕ Reject all low ({stats.low})
              </button>
              <span className="spacer" />
              <button
                className="primary"
                onClick={handleSave}
                disabled={saving || approvedCount === 0}
              >
                {saving ? "Saving…" : `Save ${approvedCount} approved`}
              </button>
            </div>
          )}

          {suggestions.length === 0 ? (
            <div className="empty">
              No edits suggested — the docs look consistent with this change, or
              try rephrasing your query.
            </div>
          ) : (
            groups.map(([file, items]) => {
              const isOpen = !collapsed.has(file);
              const high = items.filter((s) => s.confidence === "high").length;
              return (
                <div key={file} className="filegroup">
                  <button
                    className="filehead"
                    onClick={() => toggleFile(file)}
                  >
                    <span className="chevron">{isOpen ? "▾" : "▸"}</span>
                    <span className="filename">{file}</span>
                    <span className="filecount">
                      {items.length} edit{items.length > 1 ? "s" : ""}
                      {high > 0 && <span className="high"> · {high} high</span>}
                    </span>
                  </button>
                  {isOpen &&
                    items.map((s) => (
                      <SuggestionCard
                        key={s.id}
                        suggestion={s}
                        onApprove={() => handleStatus(s.id, "approve")}
                        onReject={() => handleStatus(s.id, "reject")}
                        onEdit={(text) => handleEdit(s.id, text)}
                        onRegenerate={() => handleRegenerate(s.id)}
                      />
                    ))}
                </div>
              );
            })
          )}
        </>
      )}
    </main>
  );
}
