"""
DC circuit for assembly with two voltage sources used as inputs to an op amp differential amplifier configuration.
"""

import debugsbuddy as bugbud


def run_demo():
    print('Running differential amplifier faulty circuit demo!')
    # Define the intended circuit design
    components = {'v_in': ['v1', 'v2', 'gnd', 'vcc'], #'v_out': ['vo'],
                  'res': ['r1', 'r2', 'r3', 'r4', 'rl'], 'opamp5': ['u1']}
    prms = {'r1-r': 1, 'r2-r': 1, 'r3-r': 4, 'r4-r': 4, 'rl-r': 20,
            'u1-g': 10, 'u1-ri': 100, 'u1-ro': 1}
    #correct_conns = [{'v1', 'r1.1'}, {'r1.2', 'u1.-'}, {'v2', 'r2.1'}, {'r2.2', 'u1.+'},
    #                 {'u1.-', 'r3.1'}, {'r3.2', 'vo'}, {'u1.+', 'r4.1'}, {'r4.2', 'gnd'},
    #                 {'gnd', 'rl.1'}, {'vo', 'rl.2'}, {'vo', 'u1.o'}, {'u1.vcc', 'vcc'}, {'u1.vee', 'gnd'}]
    correct_conns = [{'v1', 'r1.1'}, {'r1.2', 'u1.-'}, {'v2', 'r2.1'}, {'r2.2', 'u1.+'},
                     {'u1.-', 'r3.1'}, {'r3.2', 'u1.o'}, {'u1.+', 'r4.1'}, {'r4.2', 'gnd'},
                     {'gnd', 'rl.1'}, {'u1.o', 'rl.2'}, {'u1.vcc', 'vcc'}, {'u1.vee', 'gnd'}]
    # For now circuit is actually assembled correctly
    faulty_conns = correct_conns.copy()
    #faulty_conns.remove({'u1.+', 'r4.1'})
    meas_nodes = ['u1.o', 'u1.-', 'u1.+']

    circ = bugbud.FaultyCircuit(components, faulty_conns, correct_conns, prms, meas_nodes)
    outs = circ.simulate_test([0.3, 0.6, 0, 2])
    print(outs)

    bugbud.guided_debug(circ, vcc=True)


def run_simplified_demo():
    print('Running differential amplifier faulty circuit demo!')
    # Define the intended circuit design
    components = {'v_in': ['v1', 'v2', 'gnd', 'vcc'], #'v_out': ['vo'],
                  'res': ['r1', 'r2', 'r3', 'r4', 'rl'], 'opamp3': ['u1']}
    prms = {'r1-r': 10, 'r2-r': 10, 'r3-r': 24, 'r4-r': 24, 'rl-r': 100,
            'u1-g': 100, 'u1-ri': 1000, 'u1-ro': 1}
    #correct_conns = [{'v1', 'r1.1'}, {'r1.2', 'u1.-'}, {'v2', 'r2.1'}, {'r2.2', 'u1.+'},
    #                 {'u1.-', 'r3.1'}, {'r3.2', 'vo'}, {'u1.+', 'r4.1'}, {'r4.2', 'gnd'},
    #                 {'gnd', 'rl.1'}, {'vo', 'rl.2'}, {'vo', 'u1.o'}]
    correct_conns = [{'v1', 'r1.1'}, {'r1.2', 'u1.-'}, {'v2', 'r2.1'}, {'r2.2', 'u1.+'},
                     {'u1.-', 'r3.1'}, {'r3.2', 'u1.o'}, {'u1.+', 'r4.1'}, {'r4.2', 'gnd'},
                     {'gnd', 'rl.1'}, {'u1.o', 'rl.2'}]
    # For now circuit is actually assembled correctly
    faulty_conns = correct_conns.copy()
    faulty_conns.remove({'u1.+', 'r4.1'})
    meas_nodes = ['u1.o', 'u1.-', 'u1.+']

    circ = bugbud.FaultyCircuit(components, faulty_conns, correct_conns, prms, meas_nodes)
    #outs = circ.simulate_test([0.3, 0.6, 0, 2])
    #print(outs)

    bugbud.guided_debug(circ, mode='live', vcc=True)


if __name__ == '__main__':
    run_simplified_demo()

