export async function rewriteQueryWithOllama({
  baseUrl,
  model,
  query,
  signal
}) {
  const response = await fetch(`${baseUrl}/api/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      prompt: `Rewrite this search query for a documentation search engine. Return only the rewritten query.\n\nQuery: ${query}`,
      stream: false,
      options: {
        temperature: 0.2
      }
    }),
    signal
  });

  if (!response.ok) {
    const text = await response.text();
    return { rewrittenQuery: query, skipped: `ollama_error:${response.status}`, detail: text };
  }

  const data = await response.json();
  const content = data.response?.trim();

  if (!content) {
    return { rewrittenQuery: query, skipped: "empty_completion" };
  }

  return { rewrittenQuery: content, skipped: null };
}
