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
    outputs = [l1.p1, c1.p1]
    rlc_circ = Circuit([r1, l1, c1, v1, gnd], intended_conns, outputs)

    bugbud.guided_debug(rlc_circ, analysis='ac', mode='live')


if __name__ == '__main__':
    run_demo()
