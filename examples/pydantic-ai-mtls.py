# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic-ai-slim[openai]==2.44.0", "httpx2==2.13.0"]
# ///
"""IP-based mTLS client. See docs/private-tls.md. Never prints prompt/response text."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import ssl
from pathlib import Path
from urllib.parse import urlsplit

import httpx2
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


async def run(args: argparse.Namespace) -> None:
    ctx = ssl.create_default_context(cafile=str(Path(args.ca).expanduser()))
    ctx.load_cert_chain(Path(args.cert).expanduser(), Path(args.key).expanduser())
    api_key = Path(args.api_key_file).expanduser().read_text().strip()
    # Hidden input avoids putting a real prompt into shell history or stdout.
    prompt = getpass.getpass("Prompt (hidden): ")
    async with httpx2.AsyncClient(verify=ctx, trust_env=False, timeout=120.0) as client:
        provider = OpenAIProvider(
            base_url=args.base_url, api_key=api_key, http_client=client
        )
        agent = Agent(OpenAIChatModel(args.model, provider=provider))
        agent.instrument = False
        result = await agent.run(prompt)
        # Use result.output in memory in your application; do not log it.
        usage = result.usage
        print(
            json.dumps(
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="https://IP:port/v1")
    parser.add_argument("--model", required=True, help="MTPLX served model ID")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--api-key-file", required=True)
    args = parser.parse_args()
    if urlsplit(args.base_url).scheme != "https":
        parser.error("--base-url must use https")
    try:
        asyncio.run(run(args))
    except Exception:  # noqa: BLE001 - provider errors can contain response bodies
        # Provider/HTTP errors can include response bodies. Keep them out of logs.
        raise SystemExit(
            "Request failed; check certificates, IP, model ID and API key."
        ) from None


if __name__ == "__main__":
    main()
