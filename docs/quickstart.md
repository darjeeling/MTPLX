# Quickstart

```bash
brew install youssofal/mtplx/mtplx

mtplx help
mtplx doctor --summary
mtplx pull Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
mtplx inspect Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed --json
```

Homebrew is the recommended macOS path. Python-only installs can use PyPI:

```bash
python3 -m pip install -U mtplx
```

The built-in downloader is the default. For faster parallel downloads, install aria2 and opt in:

```bash
brew install aria2
mtplx pull --download-backend aria2 Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed
```

`mtplx pull --download-backend aria2` requires aria2c and fails if it is missing; `--download-backend auto` uses aria2c when it is installed and the built-in downloader otherwise; `python` (the default) never touches aria2c.
Hugging Face credentials are passed to aria2c through standard input and are never placed in the process arguments.

**Behind a company proxy that inspects HTTPS.** If a download stops with `CERTIFICATE_VERIFY_FAILED` while `curl` reaches the same address, the proxy signs traffic with its own root certificate. macOS and `curl` trust it through the keychain; Python does not read the keychain. Either of these fixes it:

```bash
# 1. Let Python use the macOS keychain (what pip itself does). MTPLX picks it up when it is installed.
python3 -m pip install truststore

# 2. Or point Python at the proxy's root certificate (ask your IT team for the .pem file).
export SSL_CERT_FILE=/path/to/proxy-root.pem REQUESTS_CA_BUNDLE=/path/to/proxy-root.pem
```

`MTPLX_SYSTEM_TRUST=0` keeps MTPLX on the bundled certificates even when `truststore` is installed.

The GitHub release wheel remains available for reproducible installs:

```bash
gh release download --repo youssofal/mtplx --pattern '*.whl'   # latest tagged release
python3 -m pip install ./mtplx-*-py3-none-any.whl
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
mtplx start
mtplx start cli
mtplx start cli --no-mtp
mtplx quickstart --port 8000 --no-stats-footer
```

`--no-mtp` switches generation to target-only AR. For MTP-equipped models the
MTP runtime stays loaded, so terminal chat can use `/mtp off`, `/mtp on`, and
`/mtp status` without reloading. Native AR-only models such as
`mlx-community/Laguna-S-2.1-oQ4e` instead install an unloaded AR route at
construction because there is no MTP head to retain.

For scheduler selection and backend-specific concurrent implementations, see
[Concurrency modes](concurrency.md).

The Laguna download is pinned automatically. It needs about 64.13 GB of disk
space, and the runtime's admission gate requires ≈85.3 GiB of unified memory
(weights plus runtime headroom and a 16 GiB system reserve) — in practice a
96 GB Mac, with 128 GB comfortable. Its default
context and maximum response are 32,768 tokens. A larger explicit server
context is accepted only when it fits the active Metal resident-memory cap.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
