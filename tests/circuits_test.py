from debugsbuddy.circuits import Node, Circuit
from debugsbuddy.components import *

# TODO: Write these tests properly
def test_node():
    n = Node('n', 'prnt', lambda x: x + 1)
    assert n.name is 'n'
    assert n.lims is None
    assert n.prnt_comp == 'prnt'
    assert n.calc_coeff(2) == 3
    n.lims = [2, 5]
    assert n.lims[1] == 5


def test_circuit():
    r1 = Resistor(51)
    r2 = Resistor(72)
    r3 = Resistor(89, 'r4')
    test_circ = Circuit([r1, r2, r3], [(r1.p1, r2.p2)], [r2.p1])
    assert test_circ.name is None
