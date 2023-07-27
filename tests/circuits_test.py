from debugsbuddy.circuits import Node, Circuit, Resistor


def test_circuit():
    r1 = Resistor(51)
    r2 = Resistor(72)
    r3 = Resistor(89, 'r4')
    test_circ = Circuit([r1, r2, r3], [(r1.p1, r2.p2)], [r2.p1])
    assert test_circ.name is None
