import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tara_h3'))
import minimax_h3_temporal_bootstrap as boot


def test_manifest_pins_every_source_file():
    paths = {s['path'] for s in boot.MANIFEST['sources']}
    assert 'comfy_kitchen/backends/cuda/native_temporal/build.py' in paths
    assert all(len(s['sha256']) == 64 and s['bytes'] > 0 for s in boot.MANIFEST['sources'])


def test_local_build_needs_nvcc(monkeypatch):
    for name in ('CUDA_HOME', 'CUDA_PATH', 'TARA_H3_TEMPORAL_LOCAL_BUILD'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(boot.shutil, 'which', lambda name: None)
    assert 'nvcc' in boot.local_toolchain()


def test_local_build_can_be_disabled(monkeypatch):
    monkeypatch.setenv('TARA_H3_TEMPORAL_LOCAL_BUILD', '0')
    assert 'disabled' in boot.local_toolchain()


def test_ensure_builds_locally_without_a_prebuilt(monkeypatch, tmp_path):
    monkeypatch.setenv('TARA_H3_TEMPORAL_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(boot, 'compatible', lambda device: (True, None))
    monkeypatch.setattr(boot, 'binary_for_device', lambda device: None)
    api = type('Api', (), {'is_available': staticmethod(lambda device: True)})
    monkeypatch.setattr(boot, 'build_locally', lambda device: api)
    boot._READY.clear(); boot._FAILED.clear()
    assert boot.ensure('cuda:9') is api
