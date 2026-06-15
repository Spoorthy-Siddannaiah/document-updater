// Call the backend directly (not via the Next dev rewrite proxy, which cuts
// long-running requests at ~30s). Suggestion generation can exceed that on
// cross-cutting changes. Empty default keeps the relative/proxy path for prod.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export type Confidence = "high" | "medium" | "low";
export type SuggestionStatus = "pending" | "approved" | "rejected" | "saved";

export interface Suggestion {
  id: string;
  chunk_id: string;
  file: string;
  breadcrumb: string;
  original_text: string;
  suggested_text: string;
  reason: string;
  confidence: Confidence;
  similarity: number;
  status: SuggestionStatus;
  start_line: number;
  end_line: number;
}

export interface QueryResponse {
  session_id: string;
  query: string;
  suggestions: Suggestion[];
  considered_chunks: number;
  elapsed_ms: number;
}

export interface SaveResponse {
  saved: Suggestion[];
  files: string[];
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function postQuery(query: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return jsonOrThrow<QueryResponse>(res);
}

export async function setStatus(
  sessionId: string,
  suggestionId: string,
  action: "approve" | "reject"
): Promise<Suggestion> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/suggestions/${suggestionId}/${action}`,
    { method: "POST" }
  );
  return jsonOrThrow<Suggestion>(res);
}

export async function editSuggestion(
  sessionId: string,
  suggestionId: string,
  suggestedText: string
): Promise<Suggestion> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/suggestions/${suggestionId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ suggested_text: suggestedText }),
  });
  return jsonOrThrow<Suggestion>(res);
}

export async function regenerateSuggestion(
  sessionId: string,
  suggestionId: string
): Promise<Suggestion> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/suggestions/${suggestionId}/regenerate`,
    { method: "POST" }
  );
  return jsonOrThrow<Suggestion>(res);
}

export async function saveSession(sessionId: string): Promise<SaveResponse> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/save`, { method: "POST" });
  return jsonOrThrow<SaveResponse>(res);
}
