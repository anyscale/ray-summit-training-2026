"""Query the LLM endpoint with the OpenAI client.

`build_openai_app` serves the standard OpenAI REST surface, so any OpenAI client
works: point its base_url at the Serve ingress and use any non-empty api_key
(Serve does not check it locally). This is what the notebooks query with, and
what `../engine/load_test.py` drives under load.

Run after `serve run app:build_app`:
    python query.py
"""

from openai import OpenAI

# Local Serve ingress. On an Anyscale Service, use the Service base URL + the
# bearer token; the OpenAI client takes that token as api_key.
BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_ID = "qwen-0.5b"


def make_client(base_url: str = BASE_URL, api_key: str = "fake-key") -> OpenAI:
    """An OpenAI client pointed at the Serve endpoint."""
    return OpenAI(base_url=base_url, api_key=api_key)


def chat_once(client: OpenAI, prompt: str, model: str = DEFAULT_MODEL_ID) -> str:
    """One non-streaming chat completion; returns the assistant text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
    )
    return response.choices[0].message.content


def chat_stream(client: OpenAI, prompt: str, model: str = DEFAULT_MODEL_ID):
    """Stream a chat completion token by token.

    Streaming is how you observe time-to-first-token (TTFT): the first chunk
    arrives after prefill, the rest as decode produces each token.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


if __name__ == "__main__":
    client = make_client()
    print("chat:", chat_once(client, "Name three uses for Ray Serve."))
    print("stream: ", end="", flush=True)
    for token in chat_stream(client, "Count to five."):
        print(token, end="", flush=True)
    print()
