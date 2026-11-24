import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tara_h3'))
import minimax_h3_native_temporal as node


def make(strength=2.0, n=16):
    ctx = SimpleNamespace(times=torch.zeros(1, n), strength=strength, window=0.2, falloff=0.25, cap=6.0)
    q = torch.zeros(1, 2, n, 128)
    return ctx, q


def call(override, q, original=lambda *a, **k: 'native'):
    return override(original, q, q, q, 2, skip_reshape=True, skip_output_reshape=True, transformer_options={'block_index': 3})


def test_auto_prefers_kitchen_one_pass(monkeypatch):
    ctx, q = make()
    monkeypatch.setattr(node.sol.native_kernel, 'run', lambda kind, *a, **k: 'kitchen' if kind == 'dense' else None)
    assert call(node.attention_override(ctx, backend=node.AUTO, plan=object()), q) == 'kitchen'


def test_auto_falls_back_to_partitions_once(monkeypatch):
    ctx, q = make()
    calls = []
    monkeypatch.setattr(node.sol.native_kernel, 'run', lambda *a, **k: calls.append(1))
    monkeypatch.setattr(node, 'fused', None)
    monkeypatch.setattr(node.fast, 'attention', lambda *a, **k: 'partitions')
    failed = set()
    override = node.attention_override(ctx, backend=node.AUTO, plan=object(), failed=failed)
    assert call(override, q) == 'partitions' and call(override, q) == 'partitions'
    assert failed == {'kitchen'} and len(calls) == 1


def test_zero_strength_runs_the_selected_backend_unchanged(monkeypatch):
    ctx, q = make(strength=0.0)
    monkeypatch.setattr(node.sol.native_kernel, 'run', lambda *a, **k: 'kitchen')
    assert call(node.attention_override(ctx, backend=node.AUTO, plan=object()), q) == 'native'
