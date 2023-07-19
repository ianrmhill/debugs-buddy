"""
Simple 2nd-order AC circuit with a resistor, inductor, and capacitor in series configuration.
"""

import debugsbuddy as bugbud


def run_demo():
    components = {bugbud.Resistor('r1', 10),
                  bugbud.Capacitor('c1', 0.02),
                  bugbud.Inductor('l1', 0.5),
                  bugbud.VoltSource('v1')}
    connections = [{''}]
    outputs = {'l1.1', 'c1.1'}
    rlc_circ = bugbud.FaultyCircuit()


def run_demo_alt():
    r1 = bugbud.Resistor(10)
    l1 = bugbud.Inductor(0.5)
    c1 = bugbud.Capacitor(0.02)
    v1 = bugbud.VoltSource()
    gnd = bugbud.Ground()
    intended_conns = {{v1, r1.p1}, {r1.p2, l1.p1}, {l1.p2, c1.p1}, {c1.p2, gnd}}
    outputs = {l1.p1, c1.p1}
    rlc_circ = bugbud.Circuit({r1, l1, c1, v1, gnd}, intended_conns, outputs)

    bugbud.guided_debug(rlc_circ, analysis='ac', mode='live')


if __name__ == '__main__':
    run_demo()
