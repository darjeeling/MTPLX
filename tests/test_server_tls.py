"""Real socket mTLS tests use disposable certificates and never load a model."""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import socket
import ssl
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from mtplx.server_security import (
    add_server_security_args,
    server_security_argv,
    serving_url,
    tls_kwargs,
    validate_tls,
)


@pytest.fixture(scope="module")
def certificates(tmp_path_factory):
    root = tmp_path_factory.mktemp("tls")

    def openssl(*args):
        subprocess.run(["openssl", *args], cwd=root, check=True, capture_output=True)

    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "2",
        "-subj",
        "/CN=MTPLX test CA",
        "-keyout",
        "ca.key",
        "-out",
        "ca.crt",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )
    for name, usage in (("server", "serverAuth"), ("client", "clientAuth")):
        openssl(
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            f"/CN={name}",
            "-keyout",
            f"{name}.key",
            "-out",
            f"{name}.csr",
        )
        (root / f"{name}.ext").write_text(
            "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
            f"extendedKeyUsage={usage}\n"
            + ("subjectAltName=IP:127.0.0.1\n" if name == "server" else "")
        )
        openssl(
            "x509",
            "-req",
            "-in",
            f"{name}.csr",
            "-CA",
            "ca.crt",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-days",
            "2",
            "-extfile",
            f"{name}.ext",
            "-out",
            f"{name}.crt",
        )
    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "2",
        "-subj",
        "/CN=untrusted",
        "-keyout",
        "other.key",
        "-out",
        "other.crt",
    )
    return root


def security_args(root):
    parser = argparse.ArgumentParser()
    add_server_security_args(parser)
    return parser.parse_args(
        [
            "--ssl-certfile",
            str(root / "server.crt"),
            "--ssl-keyfile",
            str(root / "server.key"),
            "--ssl-ca-certs",
            str(root / "ca.crt"),
            "--ssl-require-client-cert",
            "--log-privacy",
            "metadata-only",
        ]
    )


@pytest.mark.parametrize("command", ["serve", "quickstart"])
def test_cli_forwarding_and_default_http(command, certificates):
    from mtplx.cli import build_parser
    from mtplx.server.openai import _startup_server_url, parse_args

    parser = build_parser()
    default = parser.parse_args([command])
    assert tls_kwargs(default) == {}
    assert serving_url("http://127.0.0.1:8000", default) == "http://127.0.0.1:8000"
    options = server_security_argv(security_args(certificates))
    public = parser.parse_args([command, *options])
    child = parse_args(server_security_argv(public))
    validate_tls(child)
    assert tls_kwargs(child)["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert _startup_server_url(child) == "https://127.0.0.1:8000"
    assert child.log_privacy == "metadata-only"


@pytest.mark.parametrize(
    "values",
    [
        {"ssl_certfile": "missing"},
        {"ssl_keyfile": "missing"},
        {"ssl_require_client_cert": True},
        {"ssl_certfile": "x", "ssl_keyfile": "y", "ssl_require_client_cert": True},
        {"ssl_certfile": "x", "ssl_keyfile": "y", "ssl_ca_certs": "z"},
    ],
)
def test_invalid_combinations(values):
    with pytest.raises(ValueError):
        validate_tls(SimpleNamespace(**values))


def test_mismatched_key_and_missing_file_fail_preflight(certificates):
    args = security_args(certificates)
    args.ssl_keyfile = str(certificates / "client.key")
    with pytest.raises(ValueError, match="Invalid TLS"):
        validate_tls(args)
    args.ssl_keyfile = str(certificates / "missing.key")
    with pytest.raises(ValueError, match="Invalid TLS"):
        validate_tls(args)


@pytest.fixture
def running_tls(certificates, request):
    async def app(scope, receive, send):
        if scope["path"] == "/v1/chat/completions":
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            ).encode()
        else:
            body = b'{"status":"ok"}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            lifespan="off",
            log_config=None,
            access_log=False,
            **tls_kwargs(
                security_args(certificates)
                if getattr(request, "param", True)
                else SimpleNamespace()
            ),
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


def client_context(root, identity="client", ca="ca.crt"):
    ctx = ssl.create_default_context(cafile=str(root / ca))
    if identity:
        ctx.load_cert_chain(root / f"{identity}.crt", root / f"{identity}.key")
    return ctx


def request(port, context):
    connection = http.client.HTTPSConnection(
        "127.0.0.1", port, context=context, timeout=3
    )
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["status"] == "ok"
    finally:
        connection.close()


def test_real_mtls_accepts_valid_and_rejects_missing_untrusted_and_wrong_usage(
    running_tls, certificates
):
    request(running_tls, client_context(certificates))
    for identity in (None, "other", "server"):
        with pytest.raises((ssl.SSLError, OSError, http.client.HTTPException)):
            request(running_tls, client_context(certificates, identity))
    with pytest.raises(ssl.SSLCertVerificationError):
        request(running_tls, client_context(certificates, ca="other.crt"))
    with (
        socket.create_connection(("127.0.0.1", running_tls), timeout=3) as sock,
        pytest.raises(ssl.SSLCertVerificationError),
    ):
        client_context(certificates).wrap_socket(sock, server_hostname="127.0.0.2")


def test_pydantic_ai_example_over_real_mtls(
    running_tls, certificates, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("pydantic_ai")
    import getpass
    import runpy
    from pathlib import Path

    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/pydantic-ai-mtls.py")
    )
    api_key = tmp_path / "api-key"
    api_key.write_text("test")
    monkeypatch.setattr(getpass, "getpass", lambda *_: "private prompt")
    asyncio.run(
        example["run"](
            SimpleNamespace(
                base_url=f"https://127.0.0.1:{running_tls}/v1",
                model="test",
                ca=str(certificates / "ca.crt"),
                cert=str(certificates / "client.crt"),
                key=str(certificates / "client.key"),
                api_key_file=str(api_key),
            )
        )
    )
    output = capsys.readouterr()
    assert json.loads(output.out) == {"input_tokens": 3, "output_tokens": 1}
    assert "private prompt" not in output.out + output.err


def test_public_dry_run_carries_tls_and_privacy(
    monkeypatch, capsys, tmp_path, certificates
):
    from test_public_cli import _serve_dry_run_payload_for_model

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    options = server_security_argv(security_args(certificates))
    payload = _serve_dry_run_payload_for_model(monkeypatch, capsys, model_dir, options)
    assert payload["api_base_url"] == "https://127.0.0.1:8000/v1"
    assert payload["chat_url"].startswith("https://")
    for token in options:
        assert token in payload["argv"]


def test_invalid_tls_fails_before_loading_model(monkeypatch):
    from mtplx.commands import public
    from mtplx.server import openai

    def should_not_run(*args, **kwargs):
        pytest.fail("model work started before TLS validation")

    monkeypatch.setattr(openai, "ServerState", should_not_run)
    with pytest.raises(SystemExit, match="specified together"):
        openai.main(["--ssl-certfile", "missing.pem"])
    monkeypatch.setattr(public, "_resolve_runtime_options_on_args", should_not_run)
    assert public.cmd_serve_public(SimpleNamespace(ssl_certfile="missing.pem")) == 2


@pytest.mark.parametrize("running_tls", [False], indirect=True)
def test_default_localhost_still_serves_http(running_tls):
    connection = http.client.HTTPConnection("127.0.0.1", running_tls, timeout=3)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["status"] == "ok"
    finally:
        connection.close()


def test_security_options_do_not_reuse_an_existing_http_server(monkeypatch, capsys):
    from mtplx.cli import build_parser
    from mtplx.commands import public

    args = build_parser().parse_args(
        ["serve", "--yes", "--log-privacy", "metadata-only"]
    )
    args.quickstart_openwebui = True
    monkeypatch.setattr(public, "_port_is_busy", lambda *a, **k: True)
    monkeypatch.setattr(public, "_print_serve_start_banner", lambda *a: None)
    monkeypatch.setattr(
        public,
        "_http_json",
        lambda *a, **k: pytest.fail("must not reuse another server"),
    )
    assert public.cmd_serve_public(args) == 2
    assert "port is in use" in capsys.readouterr().out
