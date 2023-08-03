"""
Simple 2nd-order AC circuit with a resistor, inductor, and capacitor in series configuration.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    r1 = Resistor(10)
    l1 = Inductor(0.5)
    c1 = Capacitor(0.02)
    v1 = VoltSource(0, 1)
    gnd = Ground()
    intended_conns = [(v1, r1.p1), (r1.p2, l1.p1), (l1.p2, c1.p1), (c1.p2, gnd)]
    outputs = [c1.p1]
    rlc_circ = Circuit([r1, l1, c1, v1, gnd], intended_conns, outputs)

    # Set the faulty circuit parameters
    #faulty_conns = [(v1, r1.p1), (r1.p2, l1.p1), (l1.p2, gnd), (r1.p2, c1.p1), (c1.p2, gnd)]
    # TODO: With so many possible faults and only one node measured it's hard to pick out the right fault from so many!
    faulty_conns = [(v1, r1.p1), (r1.p2, l1.p1), (l1.p2, c1.p1), (c1.p2, gnd)]
    faulty_prms = {'r1': {'r': 10},
                  'l1': {'l': 0.5},
                  'c1': {'c': 0.02}}
    rlc_circ.set_actual_circuit(faulty_conns, faulty_prms)

    bugbud.guided_debug(rlc_circ, mode='simulated')


if __name__ == '__main__':
    run_demo()
