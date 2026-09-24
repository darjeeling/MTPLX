"""Shared, model-independent serving TLS options and preflight."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any


def add_server_security_args(parser: Any) -> None:
    parser.add_argument(
        "--ssl-certfile", help="PEM server certificate/chain; enables HTTPS"
    )
    parser.add_argument("--ssl-keyfile", help="PEM server private key (unencrypted)")
    parser.add_argument(
        "--ssl-ca-certs", help="PEM CA bundle for client certificate verification"
    )
    parser.add_argument(
        "--ssl-require-client-cert",
        action="store_true",
        help="Require a trusted client certificate (mTLS)",
    )
    parser.add_argument(
        "--log-privacy",
        choices=["metadata-only"],
        default=None,
        help="Keep numeric usage logs; disable content logs, captures and traces. Caches are unchanged.",
    )


def server_security_argv(args: Any) -> list[str]:
    result = []
    for name in ("ssl_certfile", "ssl_keyfile", "ssl_ca_certs", "log_privacy"):
        value = getattr(args, name, None)
        if value:
            result.extend(["--" + name.replace("_", "-"), str(value)])
    if getattr(args, "ssl_require_client_cert", False):
        result.append("--ssl-require-client-cert")
    return result


def tls_kwargs(args: Any) -> dict[str, Any]:
    cert = getattr(args, "ssl_certfile", None)
    key = getattr(args, "ssl_keyfile", None)
    ca = getattr(args, "ssl_ca_certs", None)
    required = bool(getattr(args, "ssl_require_client_cert", False))
    if bool(cert) != bool(key):
        raise ValueError("--ssl-certfile and --ssl-keyfile must be specified together")
    if (ca or required) and not cert:
        raise ValueError(
            "Client certificate verification requires --ssl-certfile and --ssl-keyfile"
        )
    if required and not ca:
        raise ValueError("--ssl-require-client-cert requires --ssl-ca-certs")
    if ca and not required:
        raise ValueError("--ssl-ca-certs requires --ssl-require-client-cert")
    if not cert:
        return {}
    return {
        "ssl_certfile": str(Path(cert).expanduser()),
        "ssl_keyfile": str(Path(key).expanduser()),
        "ssl_ca_certs": str(Path(ca).expanduser()) if ca else None,
        "ssl_cert_reqs": ssl.CERT_REQUIRED if required else ssl.CERT_NONE,
    }


def validate_tls(args: Any) -> None:
    options = tls_kwargs(args)
    if not options:
        return
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # Avoid an interactive OpenSSL password prompt during service startup.
        context.load_cert_chain(
            options["ssl_certfile"], options["ssl_keyfile"], password=lambda: ""
        )
        if options["ssl_ca_certs"]:
            context.load_verify_locations(cafile=options["ssl_ca_certs"])
        context.verify_mode = options["ssl_cert_reqs"]
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Invalid TLS certificate/key/CA configuration ({type(exc).__name__})"
        ) from None


def serving_url(url: str | None, args: Any) -> str | None:
    if url and getattr(args, "ssl_certfile", None) and url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url
