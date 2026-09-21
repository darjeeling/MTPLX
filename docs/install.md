# Install

See [INSTALL.md](../INSTALL.md) for the short path.

MTPLX is Apple-Silicon-first:

- macOS 14.0 or newer
- native arm64 Python 3.11 or newer
- `python3 -m pip install mlx` in that same environment
- enough unified memory and disk for the selected model/profile, checked by `mtplx doctor`

On modern chips, the app and CLI offer 4B Speed below 16 GB, Bonsai 2 from
16 to under 32 GB, Qwen 3.8 27B Optimized Speed from 32 to under 256 GB, and
Flash-Next Optimized Quality from 256 GB. Flash-Next options are listed from
96 GB and filtered by their catalog peak. M1/M2 keep the FP16 policy: 9B
below 32 GB when it fits, then the 27B trio. An 8 GB M1/M2 Mac has no fitting
curated FP16 model. Explicit model selections take precedence.

The two new packs need engine 2.11.4. Their final uploads and memory
qualification are separate release gates. Catalog feasibility uses peak ×
1.5 for the Recommended badge and the unchanged disk-space rule; it does not
change engine memory limits, context windows, or runtime admission checks.

The quantized 27B and 9B flagships (the Qwen 3.8 trio, Optimized-Speed, Optimized-Quality, the legacy Optimized hybrid, and their FP16 siblings, plus the 9B Speed pair) launch on the Turbo profile by default — the same NAX verify-kernel + compiled-verify fast path the macOS app uses; every other model defaults to Sustained (`--profile sustained`). `stable` remains available as the conservative compatibility alias, and Burst is available explicitly as `--profile performance-cold --max` for short-context benchmark runs.

Do not install model weights into the source checkout. Use the MTPLX model cache or a Hugging Face cache.
