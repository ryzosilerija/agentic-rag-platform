"""M0 smoke test: verifies provider config and makes one chat completion call."""
from __future__ import annotations
import sys
from src.config import settings
from src.llm.factory import get_chat_client, get_judge_model_name, get_model_name


def main() -> int:
    print("=" * 50)
    print(" Agentic RAG Platform - M0 smoke test")
    print("=" * 50)
    print(f"Provider:     {settings.provider}")
    print(f"Chat model:   {get_model_name()}")
    print(f"Judge model:  {get_judge_model_name()}")
    print()
    if settings.provider == "gemini" and not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY is empty.")
        print("  1. Get a key: https://aistudio.google.com/app/apikey")
        print("  2. Add it to .env as GEMINI_API_KEY=...")
        return 1
    client = get_chat_client()
    print("Calling model...")
    resp = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": "Respond with exactly: M0 smoke test passed"}],
        max_tokens=20,
    )
    reply = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    print(f"Response:     {reply}")
    if usage:
        print(f"Tokens used:  prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
    print()
    print("M0 smoke test complete - provider is reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
