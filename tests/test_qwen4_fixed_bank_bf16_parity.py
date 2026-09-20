"""The fixed QSA bank against the stock cache in bfloat16, bit for bit.

bfloat16 weights and hidden states (the served dtype), CPU as the parity surface like the other QSA cache tests.
The stock ``QSACache`` lane is the oracle throughout: promotion, verify-width steps that complete a pooled block,
steps that do not, a rejected window that is rolled back, growth, and the way back out of the fixed bank.
"""

import mlx.core as mx
import mlx.utils
import pytest

import mtplx.graphbank as graphbank
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs

PREFILL = 13  # odd on purpose: the first verify step straddles a block edge (ratio 2)
STEP = 4


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(3)
    layer = Attention(_tiny_args())
    layer.update(mlx.utils.tree_map(lambda p: p.astype(mx.bfloat16), layer.parameters()))
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.bfloat16)


def _promote(attn, cache):
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache, reserve_tokens=STEP, initial_reserve_tokens=32
    )
    assert promoted == 1 and failures == {}
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    return cache[0]


def _same(a: mx.array, b: mx.array) -> bool:
    return a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def test_every_step_matches_the_stock_lane_bit_for_bit(attn):
    """Steps that complete a block, steps that do not, and single rows."""

    steps = [_hidden(n, 10 + i) for i, n in enumerate((STEP, STEP, 1, 3, STEP, 1, 1, STEP))]
    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    stock = QSACache(compress_ratio=attn.indexer.ratio)
    x_pre = _hidden(PREFILL, 2)
    attn(x_pre, fixed[0])
    attn(x_pre, stock)
    _promote(attn, fixed)
    for x in steps:
        out_fixed = attn(x, fixed[0])
        out_stock = attn(x, stock)
        assert _same(out_fixed, out_stock)
    valid = fixed[0].size() // attn.indexer.ratio
    assert _same(
        fixed[0].pooled[:, :valid].astype(mx.float32),
        stock.pooled[:, :valid].astype(mx.float32),
    )


def test_a_rejected_window_leaves_no_trace(attn):
    """A block the rejected tokens completed is rewritten by the accepted ones."""

    x_pre = _hidden(PREFILL, 4)
    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, fixed[0])
    _promote(attn, fixed)
    attn(_hidden(STEP, 5), fixed[0])  # drafted window
    assert fixed[0].trim(3) == 3  # one token accepted, three rejected
    x_next = _hidden(STEP, 6)
    out = attn(x_next, fixed[0])

    stock = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, stock)
    attn(_hidden(STEP, 5)[:, :1], stock)
    golden = attn(x_next, stock)
    assert _same(out, golden)


def test_the_way_back_out_of_the_bank_is_exact(attn):
    x_pre = _hidden(PREFILL, 7)
    fixed = [QSACache(compress_ratio=attn.indexer.ratio)]
    stock = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, fixed[0])
    attn(x_pre, stock)
    bank = _promote(attn, fixed)
    for seed in (8, 9):
        x = _hidden(STEP, seed)
        attn(x, bank)
        attn(x, stock)
    entry = bank.demote()
    assert isinstance(entry, QSACache)
    assert entry.pooled.dtype == stock.pooled.dtype == mx.bfloat16
    valid = entry.pooled_len
    assert valid == stock.pooled_len
    assert _same(entry.pooled[:, :valid], stock.pooled[:, :valid])
    x_after = _hidden(STEP, 11)
    assert _same(attn(x_after, entry), attn(x_after, stock))
