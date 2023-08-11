"""
DC circuit for assembly with two voltage sources with source resistors in parallel connected to an output node with
a load resistance to GND.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    print('Running three resistor faulty circuit demo!')
    # Define the intended circuit design
    r1 = Resistor(1.2)
    r2 = Resistor(2.4)
    r3 = Resistor(1.6)
    v1 = VoltSource(0, 1)
    v2 = VoltSource(0, 1)
    gnd = Ground()
    intended_conns = [(v1, r1.p1), (v2, r2.p1), (r1.p2, r3.p1), (r2.p2, r3.p1), (r1.p2, r2.p2), (r3.p2, gnd)]
    outputs = [r3.p1, r2.p1]
    three_res_circ = Circuit([r1, r2, r3, v1, v2, gnd], intended_conns, outputs)

    faulty_conns = [(gnd, r1.p1), (v2, r2.p1), (r1.p2, r3.p1), (r2.p2, r3.p1), (r3.p2, gnd)]
    faulty_prms = {'r1': {'r': 1.2}, 'r2': {'r': 2.4}, 'r3': {'r': 1.6}}
    three_res_circ.set_actual_circuit(faulty_conns, faulty_prms)

    bugbud.guided_debug(three_res_circ, mode='simulated',
                        shrt_admittance=1e2, open_admittance=1e-2)


if __name__ == '__main__':
    run_demo()

