# Bonsai 2 memory and pack metadata handoff

This change adds the measurement needed before recommending Bonsai on small
Macs. It changes no planner budget or reserve, and establishes no RAM tier or
speed claim. The existing pack in `~/.mtplx/models` remains unchanged.

## Run the memory matrix

Run from this worktree, outside a sandbox that blocks Metal:

```sh
cd /Users/youssof/Projects/MTPLX-release/mtplx-rel-2114-bonsai
PY=/Users/youssof/Projects/MTPLX-release/mtplx-rel-2114-20260921/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" scripts/bonsai_memory_table.py \
  --pack "$HOME/.mtplx/models/Bonsai-3.8-27B-MTPLX-Optimized-Speed" \
  --out "$PWD/outputs/bonsai-memory-metal" \
  --classes 16 18 24 --contexts 4K 8K 16K
```

The output directory must be new. It receives `memory.json` and `memory.md`.
Add `--dry-run` to print the entire matrix and planner verdicts without
importing MLX, loading weights, or creating output. Classes and contexts accept
space-separated integers; a `K` context suffix means 1024 tokens. RAM is GiB.

There are 36 cases: three RAM classes, three prompt sizes, KV off/q8, and
zero/1024 decoded tokens. The planner gets `total_ram_bytes` directly. Refusal
does not suppress the experiment or truncate a prompt. Both the planner's
verdict and whether the complete request fits its context are in the JSON.

One child process loads the model once for each RAM class. Before loading, it
sets `mx.set_memory_limit` to the planner's engine budget and
`mx.set_wired_limit` using `mtplx serve`'s default formula: the smaller of the
engine budget, `max(4 GiB, 60% RAM)`, and 160 GiB. Bonsai has no resident-floor
override. Every case starts with a fresh request cache. The default turbo
profile and 2048-token prefill chunks are used, with fp16 auxiliary tensors.
Inherited `MTPLX_*` overrides are excluded from the child. The effective
runtime settings and actual Metal device are recorded.

The workload calls the engine's prefill helper and AR forward path. In q8 mode
the engine prefills contiguously, then repages/quantizes; the report includes
that transition's peak and checks every full-attention cache really uses q8.
The repeated prompt is tokenized locally to an exact token count. Greedy AR
decoding ignores EOS to execute the complete 1024-token allocation workload.
The draft head and vision tower are loaded and resident, although MTP
verification and image activations are not exercised. There is no populated
session bank. This is a text-memory instrument, not a complete application
acceptance run. It records no duration, latency, throughput, or copied speed
evidence from pack metadata.

The reported peak is `max(load peak, request peak)`, in bytes and GiB. MLX's
memory limit is a guideline: a completed row above the engine budget is marked
`within_engine_budget: false`. A large host with a smaller allocator limit
does not reproduce physical small-Mac pressure. The measurements inform the
planner change; physical-device confirmation and MTP/image/bank memory
coverage are still needed for an unconditional product recommendation.

Allocation exceptions retain their type and message. A failed class load
marks every row `load_failed`. A failed request records its peak and available
token progress, then marks the rest of that class `not_run_after_failure`.
It does not reuse a possibly failed lazy graph or silently reload the pack.
The next RAM class starts in a fresh process. Native worker termination is
reported as `worker_failed` for unfinished rows. Reports are checkpointed as
events arrive. Exit 0 means all workloads completed, not that all fit their
budgets; incomplete workloads return 1.

## Today's planner, before any measured change

`mtplx/memory_plan.py:273` gives a 16 GiB machine a 12 GiB engine budget.
At `:520`, the planner subtracts weights, 3 GiB runtime transients (`:54`),
and the 1 GiB session-bank floor (`:64`) before budgeting KV. The 3 GiB
allowance is anchored to a larger dense-pack receipt with approximately
15.9 GiB weights on a 128 GB M5 Max (`:49`). This is not a single 4 GiB
runtime scratch allocation: one quarter is the separately justified bank.

`tests/test_memory_plan_bonsai.py` pins exactly 8.2 GiB of total weights
(8,804,682,956 integer bytes), 65,536 fp16 KV bytes/token, and a 262,144-token
model context. This deliberately differs from the installed pack's actual
8,834,412,216 total safetensors bytes, including its MTP sidecar. The
measurement script always reads the actual pack sizes.

| RAM GiB | Engine GiB | Verdict | KV off context | q8 context |
| ---: | ---: | :--- | ---: | ---: |
| 16 | 12 | Refuse | 4096 fallback | 4096 fallback |
| 18 | 13.5 | Admit | 20480 | 36864 |
| 24 | 18 | Admit | 94208 | 172032 |
| 32 | 24 | Admit | 192512 | 262144 |

At 16 GiB, weights plus the reserve already need 12.2 GiB. Including the
minimum 4096-token KV allocation needs approximately 12.45 GiB with KV off
or 12.3375 GiB with q8. Quantizing KV alone cannot fix that refusal. The
4096 field returned for a refused plan is a fallback, never an admission.

Proposed follow-up, intentionally not implemented: retain the engine envelope
and bank floor, and replace only the runtime-transient term with a validated
family-specific function `T(context, kv_mode, profile, prefill_chunk, dtype)`.
For each tested configuration, form an upper envelope of
`max(load_peak - planned_weights, request_peak - planned_weights - planned_KV)`
over repetitions, then add a margin established from measurement variability.
Use the existing 3 GiB fallback outside the measured contexts/configurations.
Do not extrapolate a 16K text AR receipt to 262K or active MTP/image work, and
do not scale reserve by weight-bit count alone: activation geometry did not
shrink to two bits. Validate those other workloads before promoting their
rule. Apply the same chosen transient reserve to context fit, idle/steady bank
budgets, and dynamic reserve accounting (`plan_memory`,
`transient_reserve_bytes`, `bank_dynamic_ceiling`) so the bank cannot spend the
space that admission retained. No new numeric reserve or variance margin is
chosen in this task.

## Restamp into a new directory

The builder's public identity is now
`Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`, repository
`Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`, served id
`mtplx-bonsai-2-27b-optimized-speed`. `model_family: qwen3_8` describes the
trunk. The runtime quantization stamp names `prism_hadamard_qwen35`; the
config's loader-dispatch `model_type` and numeric quantization stay intact.
`min_engine_version: 2.11.4` is separate from the historical builder version
in `mtplx_version`.

After measuring, create a new pack and fill its card's RAM table:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" scripts/build_bonsai_mtplx_pack.py \
  --restamp "$HOME/.mtplx/models/Bonsai-3.8-27B-MTPLX-Optimized-Speed" \
  --out "$PWD/outputs/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed" \
  --memory-json "$PWD/outputs/bonsai-memory-metal/memory.json"
```

Weights are hardlinked where possible, otherwise copied. `--link-mode copy`
forces copies. Source metadata is never hardlinked; the license and notice
are carried verbatim. Source/output aliases, ancestor/descendant paths, and
existing destinations are refused. Unknown metadata and previous parity
verdicts are preserved, and output manifest checksums are refreshed. The
card is regenerated; unsupported old RAM guidance is not copied. A memory
report with different weight-file sizes is refused. No source-pack mutation,
download, upload, or new draft-head conversion occurs during restamping.
The existing `--stamp` mode remains available for measured parity/depth
updates on a deliberately chosen pack; it is not used on the original pack
in this task.

The card credits Prism ML and records S1's historical fp16 KL 2.6e-6 with
its actual scope: the report describes a synthetic-pack loader comparison.
It does not turn that number into a new full-pack quality verdict or silently
lift `exactness_baseline.status`. The card's speed section stays pending
unless `--speed-evidence-json` is explicitly supplied on build/restamp/stamp.
That JSON must be an object with `status: "measured"` and nonempty `rows`.
Each row supplies `hardware`, `context_tokens`, `ar_tokens_per_second`,
`mtp_tokens_per_second`, and `accepted_tokens_per_step`. Numeric values must
be finite and positive; zero accepted tokens is also valid. Both pretty JSON
and a final JSON line in a measurement log are accepted.

## Validation and scope decisions

The root `AGENTS.md` does not exist in this checkout. The supplied user rules,
S1 report, current source, installed metadata, recent git log/changelog, and
WS-7/WS-8 were read before implementation. `import mtplx` resolved to this
worktree. All edits and test artifacts are inside this worktree. No index,
commit, git configuration, tag, push, download, upload, or other worktree was
modified, and no sub-agents were used.

The pure-Python suite currently passes 75 tests:

```sh
mkdir -p outputs
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" -m pytest \
  tests/test_bonsai_memory_table.py tests/test_bonsai_pack_metadata.py \
  tests/test_memory_plan*.py -q -o addopts='' \
  --basetemp="$PWD/outputs/pytest-bonsai-manager-cpu"
```

Use a fresh `--basetemp` path to preserve previous test artifacts. The first
local attempt failed in fixture setup because the parent `outputs` directory
did not yet exist; it was created before the successful runs.

`tests/test_build_bonsai_mtplx_pack.py` cannot collect here: importing
`mlx.core` raises `ImportError: [metal::load_device] No Metal device available`.
A combined attempt with `tests/test_prism_hadamard_qwen35.py` aborted while
reimporting the failed MLX extension (exit 134). Neither module executed tests;
no skips or substitute numerical passes were added. Run the complete requested
selection outside the sandbox:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" "$PY" -m pytest \
  tests/test_bonsai_memory_table.py tests/test_bonsai_pack_metadata.py \
  tests/test_build_bonsai_mtplx_pack.py tests/test_prism_hadamard_qwen35.py \
  tests/test_memory_plan*.py -q -o addopts='' \
  --basetemp="$PWD/outputs/pytest-bonsai-manager-all"
```

The real matrix command was exercised in the sandbox. Its initial output at
`outputs/bonsai-memory-sandbox/` records 36 `load_failed` rows and no peak
figures, with the actual Metal import error for each class. That is runner
failure-path validation, not a GPU measurement. No selected test needs a
listening socket; the blocker encountered here was Metal. GPU peak results,
the two MLX test modules, physical small-Mac validation, and separate speed
measurements remain outstanding.

Base commit: `41e145e7ea431268d56e54a7902f1475f2e45bd4`, branch
`rel-2114-bonsai`. Changes remain uncommitted. `COMMIT_MESSAGE.txt` contains
the proposed commit message.
