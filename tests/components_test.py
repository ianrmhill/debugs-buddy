import torch as tc
import pytest

from debugsbuddy.components import *


def test_get_coeff():
    # Test all supported component types
    types = ['r', 'l', 'c', 'op-i', 'op-o-o', 'op-o-p', 'op-o-m']
    prm = tc.tensor(4.0)
    freq = tc.tensor(5.0)
    zero = tc.tensor(0.0)
    expected = [tc.tensor(0.25), tc.complex(zero, tc.tensor(-0.05)),
                tc.complex(zero, tc.tensor(20.0)), tc.tensor(0.25),
                tc.tensor(0.25), tc.tensor(-25000), tc.tensor(25000)]
    for i, type in enumerate(types):
        coeff = calc_coeff(type, prm, freq)
        assert tc.eq(expected[i], coeff)
    # Test default frequency args
    assert tc.eq(tc.complex(zero, tc.tensor(-25000.0)), calc_coeff('l', prm))
    # Test batched calculation
    prm = tc.tensor([[1, 2, 3], [5, 6, 7]], dtype=tc.float)
    freq = tc.tensor([[3, 4, 3], [1, 2, 1]], dtype=tc.float)
    expected = tc.tensor([[tc.complex(zero, tc.tensor(3.0)), tc.complex(zero, tc.tensor(8.0)), tc.complex(zero, tc.tensor(9.0))],
                          [tc.complex(zero, tc.tensor(5.0)), tc.complex(zero, tc.tensor(12.0)), tc.complex(zero, tc.tensor(7.0))]])
    assert tc.allclose(expected, calc_coeff('c', prm, freq))
    # Test invalid coeff type handling
    with pytest.raises(Exception):
        calc_coeff('invalid', prm, freq)
