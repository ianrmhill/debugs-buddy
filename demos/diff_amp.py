"""
DC circuit for assembly with two voltage sources used as inputs to an op amp differential amplifier configuration.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


# TODO / FIXME: This demo is too big to fit in graphics memory with the current setup, figure out solution
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
    rl = Resistor(20, 'rl')
    op = OpAmp(1000, 2, 1000, [0, 1])
    intended_conns = [(v1, r1.p1), (r1.p2, op.im), (v2, r2.p1), (r2.p2, op.ip),
                      (op.im, r3.p1), (r3.p2, op.o), (op.ip, r4.p1), (r4.p2, gnd),
                      (op.o, rl.p1), (rl.p2, gnd),
                      # The following connections are for effectively connected nodes, otherwise the fault model will
                      # complain that they are unintentionally shorted
                      # TODO: Implement automatic logic to handle this. Seems annoying to put it on the user
                      (r4.p2, rl.p2), (r1.p2, r3.p1), (r2.p2, r4.p1), (r3.p2, rl.p1)]
    outputs = [op.o, op.im, op.ip, r1.p1, r2.p1]
    diff_amp_circ = Circuit([v1, v2, gnd, r1, r2, r3, r4, rl, op], intended_conns, outputs)

    actual_conns = [(v1, r1.p1), (r1.p2, op.im), (v2, r2.p1), (r2.p2, op.ip),
                      (op.im, r3.p1), (r3.p2, op.o), (op.ip, r4.p1), (r4.p2, gnd),
                      (op.o, rl.p1), (rl.p2, gnd)]
    # TODO: Try to avoid string literals when specifying actual parameters, consider rework
    actual_prms = {'r1': {'r': 1}, 'r2': {'r': 1}, 'r3': {'r': 4}, 'r4': {'r': 4}, 'rl': {'r': 20},
                   'op1': {'rin': 1000, 'rout': 2, 'gain': 1000}}
    diff_amp_circ.set_actual_circuit(actual_conns, actual_prms)


    bugbud.guided_debug(diff_amp_circ, 'simulated')


if __name__ == '__main__':
    run_demo()

