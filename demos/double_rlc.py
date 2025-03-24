"""
Simple 2nd-order AC circuit with a resistor, inductor, and capacitor in series configuration.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    # Define the intended circuit, resonates at 0.4Hz / 2.51rad/s and 1.145Hz / 7.194rad/s
    r1 = Resistor(1)
    l1 = Inductor(0.5)
    c1 = Capacitor(0.3)
    l2 = Inductor(0.2)
    c2 = Capacitor(0.1)
    v1 = VoltSource(0, 1)
    gnd = Ground()
    intended_conns = [(v1, l1.p1), (v1, l2.p1), (l1.p2, c1.p1), (l2.p2, c2.p1), (c1.p2, r1.p1), (c2.p2, r1.p1), (r1.p2, gnd)]
    outputs = [r1.p1, l1.p2, l2.p2]
    two_pole_rlc = Circuit([r1, l1, c1, l2, c2, v1, gnd], intended_conns, outputs)
    two_pole_rlc.intended_prms ={'r1': {'r': 1}, 'l1': {'l': 0.5}, 'c1': {'c': 0.3}, 'l2': {'l': 0.2}, 'c2': {'c': 0.1}}

    # Set the faulty circuit parameters
    faulty_conns = [(v1, l1.p1), (v1, l2.p1), (l1.p2, c1.p1), (l2.p2, c2.p1), (r1.p1, c1.p2), (r1.p1, c2.p2), (r1.p2, gnd)]
    #faulty_conns.remove((l1.p2, c1.p1))
    #faulty_conns.remove((r1.p1, c1.p2))
    faulty_prms = {'r1': {'r': 1}, 'l1': {'l': 0.2}, 'c1': {'c': 0.3}, 'l2': {'l': 0.5}, 'c2': {'c': 0.1}}
    two_pole_rlc.set_actual_circuit(faulty_conns, faulty_prms)

    bugbud.guided_debug(two_pole_rlc, mode='simulated', meas_error=0.05, #shrt_admittance=1e4, open_admittance=1e-4, meas_error=0.05,
                        shrt_fault_prob=0.01, open_fault_prob=0.01, discrete_volt_steps=11, discrete_freq_steps=13)


if __name__ == '__main__':
    run_demo()
