"""
DC circuit for assembly with two voltage sources used as inputs to an op1 amp differential amplifier configuration.
"""

import debugsbuddy as bugbud
from debugsbuddy.components import *
from debugsbuddy.circuits import Circuit


def run_demo():
    print('Running two-stage amplifier circuit demo!')
    # Define the intended circuit design
    v1 = VoltSource(0, 1)
    v2 = VoltSource(0, 1)
    gnd = Ground()
    r1 = Resistor(1)
    r2 = Resistor(1)
    r3 = Resistor(4)
    r6 = Resistor(4)
    r4 = Resistor(4)
    r7 = Resistor(4)
    r5 = Resistor(10)
    r8 = Resistor(10)
    op1 = OpAmp(10, 0.2, 10, [0, 1])
    op2 = OpAmp(10, 0.2, 10, [0, 1])
    r9 = Resistor(3)
    r10 = Resistor(6)
    r11 = Resistor(10)
    intended_conns = [(v1, r1.p1), (r1.p2, op1.im), (v2, r2.p1), (r2.p2, op1.ip),
                      (r3.p1, op1.im), (r3.p2, op1.o), (r6.p1, op1.im), (r6.p2, op1.o),
                      (op1.ip, r4.p1), (r4.p2, gnd), (op1.ip, r7.p1), (r7.p2, gnd),
                      (op1.o, r5.p1), (r5.p2, r8.p1), (r8.p2, gnd),
                      (r5.p2, op2.ip), (r9.p1, op2.im), (r10.p1, op2.im), (r10.p2, op2.o), (r9.p2, gnd),
                      (r11.p1, op2.o), (r11.p2, gnd)]
    outputs = [op1.o, op1.im, op1.ip, op2.im, op2.ip, op2.o]
    diff_amp_circ = Circuit([v1, v2, gnd, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, op1, op2], intended_conns, outputs)

    # Define the implemented circuit design
    actual_conns = [(v1, r1.p1), (r1.p2, op1.im), (v2, r2.p1), (r2.p2, op1.ip),
                    (r3.p1, op1.im), (r3.p2, op1.o), (r6.p1, op1.im), (r6.p2, op1.o),
                    (op1.ip, r4.p1), (r4.p2, gnd), (op1.ip, r7.p1), (r7.p2, gnd),
                    (op1.o, r5.p1), (r5.p2, r8.p1), (r8.p2, gnd),
                    (r5.p2, op2.ip), (r9.p1, op2.im), (r10.p1, op2.im), (r10.p2, op2.o), (r9.p2, gnd),
                    (r11.p1, op2.o), (r11.p2, op2.ip)]
    actual_prms = {'r1': {'r': 1}, 'r2': {'r': 1}, 'r3': {'r': 4}, 'r4': {'r': 4}, 'r5': {'r': 10},
                   'r6': {'r': 4}, 'r7': {'r': 4}, 'r8': {'r': 10}, 'r9': {'r': 3}, 'r10': {'r': 6}, 'r11': {'r': 10},
                   'op1': {'rin': 10, 'rout': 0.2, 'gain': 10},
                   'op2': {'rin': 10, 'rout': 0.2, 'gain': 10},
                   }
    diff_amp_circ.set_actual_circuit(actual_conns, actual_prms)

    # Run the automated debug tool
    bugbud.guided_debug(diff_amp_circ, 'simulated', shrt_fault_prob=0.01, op1en_fault_prob=0.01, meas_error=0.04,
                        discrete_volt_steps=11, eig_samples=6000)


if __name__ == '__main__':
    run_demo()

