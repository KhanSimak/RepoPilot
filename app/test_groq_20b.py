import asyncio
from groq import AsyncGroq
from app.config import get_settings

settings = get_settings()

async def main():
    print("MODEL:", settings.groq_model)

    client = AsyncGroq(api_key=settings.groq_api_key)

    response = await client.chat.completions.create(
        model=settings.groq_model,
        response_format={"type": "json_object"},
        max_completion_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": """
Return ONLY valid JSON.

Schema:
{
  "thought": "brief reasoning",
  "action": "retrieve|expand_graph|answer",
  "action_input": "string",
  "missing_information": []
}
"""
            },
            {
                "role": "user",
                "content": """
Question: Where is the Flask application initialized?

Return the JSON decision only.
"""
            }
        ],
    )

    print(response.choices[0].message.content)

asyncio.run(main())