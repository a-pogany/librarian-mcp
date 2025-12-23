export async function rewriteQueryWithOpenAI({
  apiKey,
  model,
  baseUrl,
  query,
  signal
}) {
  if (!apiKey) {
    return { rewrittenQuery: query, skipped: "missing_api_key" };
  }

  const response = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content:
            "You rewrite search queries for a documentation search engine. Return only the rewritten query, no extra text."
        },
        {
          role: "user",
          content: query
        }
      ]
    }),
    signal
  });

  if (!response.ok) {
    const text = await response.text();
    return { rewrittenQuery: query, skipped: `openai_error:${response.status}`, detail: text };
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content?.trim();

  if (!content) {
    return { rewrittenQuery: query, skipped: "empty_completion" };
  }

  return { rewrittenQuery: content, skipped: null };
}
