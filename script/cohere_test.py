import os
import cohere

API_KEY = os.getenv("COHERE_API_KEY")
MODEL = "rerank-v4.0-fast"

if not API_KEY:
    raise RuntimeError("COHERE_API_KEY is not set.")

client = cohere.ClientV2(api_key=API_KEY)

query = "How does the login endpoint validate the request?"

documents = [
    "The login endpoint gets JSON data, checks whether the payload exists, then validates email and password.",
    "The application creates a Qdrant collection with 384-dimensional vectors.",
    "The login endpoint verifies the user's password with bcrypt.",
    "Redis is used for caching query results.",
    "The Flask application registers the auth blueprint.",
]

response = client.rerank(
    model=MODEL,
    query=query,
    documents=documents,
    top_n=3,
)

print(f"Model: {MODEL}")
print(f"Results: {len(response.results)}")
print()

for result in response.results:
    print(f"index={result.index}")
    print(f"score={result.relevance_score:.6f}")
    print(f"document={documents[result.index]}")
    print("-" * 60)

assert response.results, "Cohere returned no reranking results."

for result in response.results:
    assert 0.0 <= result.relevance_score <= 1.0
    assert 0 <= result.index < len(documents)

scores = [r.relevance_score for r in response.results]
assert scores == sorted(scores, reverse=True), "Results are not sorted by score."

print("\nCOHERE RERANK TEST PASSED")