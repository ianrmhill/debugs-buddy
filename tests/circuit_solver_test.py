import pytest
import torch as tc

from debugsbuddy.circuit_solver import calc_coeff, solve_circuit_complex, Node

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

def test_calc_coeff():
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


def test_solve_circuit():
    source_vals = tc.tensor([2.0], device=pu)
    frequency = tc.tensor(0.5, device=pu)
    zero = tc.tensor(0.0, device=pu)
    nodes = [Node('gnd', 'g1', ['self', '', '', '', '', '']), Node('v_in', 'v1', ['', 'self', '', '', '', '']),
             Node('r', 'r1', ['', '', 'self', 'r', '', '']), Node('r', 'r1', ['', '', 'r', 'self', '', '']),
             Node('r', 'r2', ['', '', '', '', 'self', 'r']), Node('r', 'r2', ['', '', '', '', 'r', 'self'])]
    edge_states = tc.zeros((6, 6), device=pu)
    edge_states[0, 5] = 1
    edge_states[5, 0] = 1
    edge_states[1, 2] = 1
    edge_states[2, 1] = 1
    edge_states[3, 4] = 1
    edge_states[4, 3] = 1
    prms = {'r1': {'r': tc.tensor(5.0, device=pu)}, 'r2': {'r': tc.tensor(4.0, device=pu)}}

    solution = solve_circuit_complex(source_vals, frequency, nodes, edge_states, prms)
    assert tc.allclose(solution[3], tc.complex(tc.tensor([0.88837], device=pu), zero))
    assert tc.allclose(solution[5], tc.complex(tc.tensor([0.000022209], device=pu), zero))
