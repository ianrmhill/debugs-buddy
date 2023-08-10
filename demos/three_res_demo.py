"""
DC circuit for assembly with two voltage sources with source resistors in parallel connected to an output node with
a load resistance to GND.
"""

import torch as tc

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


def run_demo():
    print('Running three resistor faulty circuit demo!')
    # Define the intended circuit design
    r1 = Resistor(6.8)
    r2 = Resistor(8.2)
    r3 = Resistor(2.4)
    v1 = VoltSource(0, 1)
    v2 = VoltSource(0, 1)
    gnd = Ground()
    intended_conns = [(v1, r1.p1), (v2, r2.p1), (r1.p2, r3.p1), (r2.p2, r3.p1), (r1.p2, r2.p2), (r3.p2, gnd)]
    outputs = [r3.p1, r2.p1]
    three_res_circ = Circuit([r1, r2, r3, v1, v2, gnd], intended_conns, outputs)

    faulty_conns = [(v1, r1.p1), (v2, r2.p1), (r1.p2, r3.p1), (r2.p2, r3.p1), (r3.p2, gnd), (r1.p1, gnd)]
    faulty_prms = {'r1': {'r': 6.8}, 'r2': {'r': 8.2}, 'r3': {'r': 2.4}}
    three_res_circ.set_actual_circuit(faulty_conns, faulty_prms)

    bugbud.guided_debug(three_res_circ, mode='simulated',
                        shrt_admittance=tc.tensor(1e2, device=pu), open_admittance=tc.tensor(1e-3, device=pu))


if __name__ == '__main__':
    run_demo()

