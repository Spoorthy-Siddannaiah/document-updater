"use client";

import { useState } from "react";
import { Suggestion } from "@/lib/api";
import { lineDiff } from "@/lib/diff";

interface Props {
  suggestion: Suggestion;
  onApprove: () => void;
  onReject: () => void;
  onEdit: (text: string) => Promise<void>;
  onRegenerate: () => Promise<void>;
}

export default function SuggestionCard({
  suggestion,
  onApprove,
  onReject,
  onEdit,
  onRegenerate,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(suggestion.suggested_text);
  const [saving, setSaving] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  async function regenerate() {
    setRegenerating(true);
    try {
      await onRegenerate();
    } finally {
      setRegenerating(false);
    }
  }

  const cls = ["card", suggestion.status].join(" ");
  const rows = lineDiff(suggestion.original_text, suggestion.suggested_text);

  async function commitEdit() {
    setSaving(true);
    try {
      await onEdit(draft);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={cls}>
      <div className="card-head">
        <span className="file">
          {suggestion.file}:{suggestion.start_line}-{suggestion.end_line}
        </span>
        <span className="breadcrumb">{suggestion.breadcrumb}</span>
        <span className="spacer" />
        <span className={`badge ${suggestion.confidence}`}>
          {suggestion.confidence}
        </span>
        {suggestion.status === "saved" && (
          <span className="badge status-saved">saved</span>
        )}
      </div>

      <div className="reason">{suggestion.reason}</div>

      <div className="sidebyside">
        <div className="pane old">
          <div className="pane-label">Current section</div>
          <div className="diff">
            {rows.map((r, i) =>
              r.type === "add" ? (
                <div key={i} className="drow blank">
                  <span className="sign"> </span>
                  <span className="dtext"> </span>
                </div>
              ) : (
                <div key={i} className={`drow ${r.type === "del" ? "del" : "same"}`}>
                  <span className="sign">{r.type === "del" ? "−" : " "}</span>
                  <span className="dtext">{r.text || " "}</span>
                </div>
              )
            )}
          </div>
        </div>

        <div className="pane new">
          <div className="pane-label">
            {editing ? "Suggested update (editing)" : "Suggested update"}
          </div>
          {editing ? (
            <textarea
              className="editor"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
            />
          ) : (
            <div className="diff">
              {rows.map((r, i) =>
                r.type === "del" ? (
                  <div key={i} className="drow blank">
                    <span className="sign"> </span>
                    <span className="dtext"> </span>
                  </div>
                ) : (
                  <div key={i} className={`drow ${r.type === "add" ? "add" : "same"}`}>
                    <span className="sign">{r.type === "add" ? "+" : " "}</span>
                    <span className="dtext">{r.text || " "}</span>
                  </div>
                )
              )}
            </div>
          )}
        </div>
      </div>

      <div className="card-actions">
        {editing ? (
          <>
            <button className="primary" onClick={commitEdit} disabled={saving}>
              {saving ? "Saving…" : "Save edit"}
            </button>
            <button
              onClick={() => {
                setDraft(suggestion.suggested_text);
                setEditing(false);
              }}
              disabled={saving}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <button
              className={`approve ${
                suggestion.status === "approved" || suggestion.status === "saved"
                  ? "active"
                  : ""
              }`}
              onClick={onApprove}
              disabled={suggestion.status === "saved"}
            >
              ✓ Approve
            </button>
            <button
              className={`reject ${
                suggestion.status === "rejected" ? "active" : ""
              }`}
              onClick={onReject}
              disabled={suggestion.status === "saved"}
            >
              ✕ Reject
            </button>
            <button
              onClick={() => {
                setDraft(suggestion.suggested_text);
                setEditing(true);
              }}
              disabled={suggestion.status === "saved"}
            >
              ✎ Edit
            </button>
            <button
              onClick={regenerate}
              disabled={suggestion.status === "saved" || regenerating}
            >
              {regenerating ? "↻ Regenerating…" : "↻ Regenerate"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
