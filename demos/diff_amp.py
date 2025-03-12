"""
DC circuit for assembly with two voltage sources used as inputs to an op amp differential amplifier configuration.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    print('Running differential amplifier faulty circuit demo!')
    # Define the intended circuit design
    v1 = VoltSource(0, 1)
    v2 = VoltSource(0, 1)
    gnd = Ground()
    r1 = Resistor(1)
    r2 = Resistor(1)
    r3 = Resistor(4)
    r4 = Resistor(4)
    op = OpAmp(10, 0.2, 10, [0, 1])
    intended_conns = [(v1, r1.p1), (r1.p2, op.im), (v2, r2.p1), (r2.p2, op.ip),
                      (op.im, r3.p1), (r3.p2, op.o), (op.ip, r4.p1), (r4.p2, gnd),
                      # The following connections are for effectively connected nodes, otherwise the fault model may
                      # complain that they are unintentionally shorted
                      (r1.p2, r3.p1), (r2.p2, r4.p1)]
    outputs = [op.o, op.im, op.ip]
    diff_amp_circ = Circuit([v1, v2, gnd, r1, r2, r3, r4, op], intended_conns, outputs)

    actual_conns = [(v1, r1.p1), (r1.p2, op.im), (v2, r2.p1), (r2.p2, op.ip),
                    (op.im, r3.p1), (r3.p2, op.o), (op.ip, r4.p1), (r4.p2, gnd)]
    actual_conns.append((r3.p2, r4.p2))
    actual_prms = {'r1': {'r': 1}, 'r2': {'r': 1}, 'r3': {'r': 4}, 'r4': {'r': 4},
                   'op1': {'rin': 10, 'rout': 0.2, 'gain': 10}}
    diff_amp_circ.set_actual_circuit(actual_conns, actual_prms)


    bugbud.guided_debug(diff_amp_circ, 'simulated', shrt_fault_prob=0.02, open_fault_prob=0.02, meas_error=0.04,
                        open_admit_inf=1e-4, shrt_admit_inf=1e4)


if __name__ == '__main__':
    run_demo()

