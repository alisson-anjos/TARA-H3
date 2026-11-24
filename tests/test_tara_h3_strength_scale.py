import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tara_h3'))
import minimax_h3_native_temporal as node


@pytest.mark.parametrize('rows,expected', [(252, 2.0), (180, 2.0), (299, 2.0 * (252 / 299) ** 0.5), (880, 2.0 * (252 / 880) ** 0.5)])
def test_auto_lowers_strength_for_denser_references_only(rows, expected):
    assert node.effective_strength(2.0, rows, node.STRENGTH_SCALES[1]) == pytest.approx(expected)


def test_fixed_keeps_strength():
    assert node.effective_strength(2.0, 880, node.STRENGTH_SCALES[0]) == 2.0


def test_720p_reference_lands_near_the_measured_optimum():
    assert 0.9 < node.effective_strength(2.0, 22 * 40, node.STRENGTH_SCALES[1]) < 1.2


def test_default_is_fixed():
    assert node.effective_strength(2.0, 880) == 2.0
