"""
DC circuit for assembly with two voltage sources with source resistors in parallel connected to an output node with
a load resistance to GND.
"""

import debugsbuddy as bugbud


def run_demo():
    print('Running three resistor faulty circuit demo!')
    # Define the intended circuit design
    components = {'v_in': ['v1', 'v2', 'gnd'], 'res': ['r1', 'r2', 'r3'], 'v_out': ['vo']}
    prms = {'r1-r': 68, 'r2-r': 82, 'r3-r': 24}
    intended_conns = [{'v1', 'r1.1'}, {'v2', 'r2.1'}, {'gnd', 'r3.1'}, {'vo', 'r1.2'}, {'vo', 'r2.2'}, {'vo', 'r3.2'}]
    # For now circuit is actually assembled correctly
    faulty_conns = [{'v1', 'r1.1'}, {'v2', 'r2.1'}, {'gnd', 'r3.1'}, {'v1', 'r1.2'}, {'vo', 'r2.2'}, {'vo', 'r3.2'}]
    meas_nodes = ['vo']

    circ = bugbud.FaultyCircuit(components, faulty_conns, intended_conns, prms, meas_nodes)
    #outs = circ.simulate_test([0.4, 0.8, 0])
    #print(outs)

    bugbud.guided_debug(circ, mode='live')


if __name__ == '__main__':
    run_demo()

