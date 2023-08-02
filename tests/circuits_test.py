from debugsbuddy.circuits import Node, Circuit
from debugsbuddy.components import *


def test_node():
    n = Node('n', 'prnt', 'p1', ['', 'self', 'n', ''])
    assert n.name is None
    assert n.type == 'n'
    assert n.prnt_comp == 'prnt'
    assert n.prnt_pin == 'p1'
    assert n.hard_conns == ['', 'self', 'n', '']
    n = Node('', '', '', [], 'setname')
    assert n.name == 'setname'


def test_circuit():
    r1 = Resistor(51)
    r2 = Resistor(72)
    r3 = Resistor(89, 'r4')
    test_circ = Circuit([r1, r2, r3], [(r1.p1, r2.p2)], [r2.p1])
    assert test_circ.name is None

    c1 = Capacitor(0.004)
