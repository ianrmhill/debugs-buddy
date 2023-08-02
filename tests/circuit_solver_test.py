import torch as tc

from debugsbuddy.components import *
from debugsbuddy.circuits import *
from debugsbuddy.circuit_solver import solve_circuit_complex

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


def test_solve_circuit():
    # Test most basic case where complex numbers aren't needed
    r1 = Resistor(4)
    r2 = Resistor(6)
    vin = VoltSource(0, 1)
    gnd = Ground()
    conns = [(vin, r1.p1), (r1.p2, r2.p1), (r2.p2, gnd)]
    outputs = [r2.p1]
    rdiv = Circuit([r1, r2, vin, gnd], conns, outputs)
    rdiv.set_actual_circuit(conns, {'r1': {'r': 4}, 'r2': {'r': 6}})
    v = solve_circuit_complex(tc.tensor([1]), tc.tensor(8), rdiv.nodes, rdiv.actual_conns, rdiv.actual_prms)
    assert round(float(v[1].abs()), 2) == 0.60
    assert round(float(v[2].abs()), 2) == 0.60

    # Test simulating around the corner frequency of a filter
    r = Resistor(10)
    c = Capacitor(0.002)
    conns = [(vin, r.p1), (r.p2, c.p1), (c.p2, gnd)]
    outputs = [c.p1]
    rc_circ = Circuit([r, c, vin, gnd], conns, outputs)
    rc_circ.set_actual_circuit(conns, {'r1': {'r': 10}, 'c1': {'c': 0.002}})
    v = solve_circuit_complex(tc.tensor([1]), tc.tensor(8 * 2 * 3.141592),
                              rc_circ.nodes, rc_circ.actual_conns, rc_circ.actual_prms)
    assert round(float(v[1].abs()), 4) == 0.7055
    assert round(float(v[1].angle()), 4) == -0.7844

    # Test simulating an op amp
    r2 = Resistor(10)
    opamp = OpAmp(1e5, 10, 10000)
    conns = [(vin, r1.p1), (r1.p2, opamp.im), (gnd, opamp.ip), (r2.p1, opamp.im), (r2.p2, opamp.o)]
    outputs = [opamp.o]
    inv_opamp = Circuit([r1, r2, opamp, vin, gnd], conns, outputs)
    inv_opamp.set_actual_circuit(conns, {'r1': {'r': 4}, 'r2': {'r': 8},
                                         'op1': {'rin': 1e4, 'rout': 0.5, 'gain': 10000}})
    v = solve_circuit_complex(tc.tensor([1]), tc.tensor(1),
                              inv_opamp.nodes, inv_opamp.actual_conns, inv_opamp.actual_prms)
    assert round(float(v[6].abs()), 2) == 1.99

    # Test op amp rail saturation functionality
    opamp_limd = OpAmp(1e5, 10, 10000, [-1.5, 1.5])
    conns = [(vin, r1.p1), (r1.p2, opamp_limd.im), (gnd, opamp_limd.ip), (r2.p1, opamp_limd.im), (r2.p2, opamp_limd.o)]
    outputs = [opamp_limd.o]
    limd_inv = Circuit([r1, r2, opamp_limd, vin, gnd], conns, outputs)
    limd_inv.set_actual_circuit(conns, {'r1': {'r': 4}, 'r2': {'r': 8},
                                        'op1': {'rin': 1e4, 'rout': 0.5, 'gain': 10000}})
    v = solve_circuit_complex(tc.tensor([1]), tc.tensor(1),
                              limd_inv.nodes, limd_inv.actual_conns, limd_inv.actual_prms)
    assert round(float(v[6].abs()), 2) == 1.50

    # Test batching functionality with op amp saturation in some batch elements
    conns = tc.zeros((2, 2, 9, 9), device=pu)
    conns[..., 0, 7], conns[..., 7, 0], conns[..., 1, 5], conns[..., 5, 1] = (1, 1, 1, 1)
    conns[..., 2, 5], conns[..., 5, 2], conns[..., 3, 6], conns[..., 6, 3] = (1, 1, 1, 1)
    conns[..., 4, 8], conns[..., 8, 4] = (1, 1)
    prms = {'r1': {'r': tc.tensor([[4, 4], [6, 4]], device=pu)}, 'r2': {'r': tc.tensor([[8, 8], [8, 8]], device=pu)},
            'op1': {'rin': tc.tensor([[1e4, 1e4], [1e4, 1e4]], device=pu),
                    'rout': tc.tensor([[0.5, 0.5], [0.5, 0.5]], device=pu),
                    'gain': tc.tensor([[1e4, 1e4], [1e4, 1e4]], device=pu)}}
    v = solve_circuit_complex(tc.tensor([[[1], [0.5]], [[1], [2]]]), tc.tensor([[1, 1], [1, 1]]),
                              limd_inv.nodes, conns, prms)
    assert round(float(v[0, 0, 6].abs()), 2) == 1.50
    assert round(float(v[0, 1, 6].abs()), 2) == 1.00
    assert round(float(v[1, 0, 6].abs()), 2) == 1.33
    assert round(float(v[1, 1, 6].abs()), 2) == 1.50
