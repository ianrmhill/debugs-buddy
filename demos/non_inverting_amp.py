"""
Non-inverting op amp circuit is one of the simplest DC circuits using an op amp.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    r1 = Resistor(2)
    r2 = Resistor(4)
    op1 = OpAmp(1, 0.2, 1, [0, 2])
    v1 = VoltSource(0, 1)
    gnd = Ground()
    intended_conns = [(v1, op1.ip), (r1.p1, op1.im), (r1.p2, gnd), (r2.p1, op1.im), (r2.p2, op1.o), (r1.p1, r2.p1)]
    outputs = [op1.o]
    int_amp_circ = Circuit([r1, r2, op1, v1, gnd], intended_conns, outputs)

    # Set the faulty circuit parameters
    faulty_conns = [(v1, op1.ip), (r1.p1, op1.im), (r1.p2, gnd), (r2.p1, op1.im), (r2.p2, op1.o)]
    faulty_prms = {'r1': {'r': 2},
                   'r2': {'r': 4},
                   'op1': {'rin': 1, 'rout': 0.2, 'gain': 1}}
    int_amp_circ.set_actual_circuit(faulty_conns, faulty_prms)

    bugbud.guided_debug(int_amp_circ, mode='simulated', shrt_admittance=1e2, open_admittance=1e-2)


if __name__ == '__main__':
    run_demo()
