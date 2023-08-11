"""
Op amp integrator circuit is the simplest AC circuit using an op amp. Behaves like a low-pass filter in AC operation.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    r1 = Resistor(1)
    r2 = Resistor(3)
    c1 = Capacitor(0.2)
    op1 = OpAmp(1, 0.2, 1, [-1, 1])
    v1 = VoltSource(0, 1)
    gnd = Ground()
    intended_conns = [(v1, r1.p1), (r1.p2, op1.im), (op1.ip, gnd),
                      (c1.p1, op1.im), (r2.p1, op1.im), (c1.p2, op1.o), (r2.p2, op1.o),
                      (r1.p2, c1.p1), (r1.p2, r2.p1), (c1.p1, r2.p1), (c1.p2, r2.p2)]
    outputs = [op1.o, op1.im, op1.ip]
    int_amp_circ = Circuit([r1, r2, c1, op1, v1, gnd], intended_conns, outputs)

    # Set the faulty circuit parameters
    faulty_conns = [(v1, r1.p1), (r1.p2, op1.im), (op1.ip, gnd),
                    (c1.p1, op1.im), (r2.p1, op1.im), (c1.p2, op1.o), (r2.p2, op1.o)]
    faulty_prms = {'r1': {'r': 1},
                   'r2': {'r': 3},
                   'c1': {'c': 0.2},
                   'op1': {'rin': 1, 'rout': 0.2, 'gain': 1}}
    int_amp_circ.set_actual_circuit(faulty_conns, faulty_prms)

    bugbud.guided_debug(int_amp_circ, mode='simulated', shrt_admittance=1e2, open_admittance=1e-2)


if __name__ == '__main__':
    run_demo()
