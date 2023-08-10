import torch as tc
import pytest

from debugsbuddy.components import *

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


def test_get_coeff():
    # Ground node test
    g = Ground()
    assert g.name is None
    assert g.p1.prnt_comp is None
    g = Ground('agnd')
    assert g.name == 'agnd'
    assert g.type == 'gnd'
    assert g.p1.name == 'gnd'
    assert g.p1.prnt_comp == 'agnd'
    assert g.list_pins()[0].name == 'gnd'
    with pytest.raises(Exception):
        g.get_coeff('', '', '', 2, 3)

    # Voltage source node test
    v = VoltSource()
    assert v.name is None
    assert v.range[1] is None
    assert v.p1.prnt_comp is None
    v = VoltSource(-1, 0.8, 'v1')
    assert v.name == 'v1'
    assert v.type == 'vin'
    assert v.range[1] is 0.8
    assert v.p1.name == 'vin'
    assert v.p1.prnt_comp == 'v1'
    assert v.list_pins()[0].name == 'vin'
    with pytest.raises(Exception):
        v.get_coeff('', '', '', 2, 3)

    # Resistor test
    r = Resistor(8)
    assert r.type == 'r'
    assert r.prms['r'] == 8
    assert r.name is None
    assert r.p1.name is 'p1'
    assert r.p2.prnt_comp is None
    assert r.p2.lims is None
    r = Resistor(5, 'r1')
    assert r.name == 'r1'
    assert r.p1.prnt_comp == 'r1'
    assert r.list_pins()[1].name == 'p2'
    assert r.get_coeff('p1', 'p1', 'p2', r.prms, 2) == 0.2

    # Inductor test
    l = Inductor(0.4)
    assert l.type == 'l'
    assert l.prms['l'] == 0.4
    assert l.name is None
    assert l.p1.name is 'p1'
    assert l.p2.prnt_comp is None
    assert l.p2.lims is None
    l = Inductor(0.2, 'smth')
    assert l.name == 'smth'
    assert l.p1.prnt_comp == 'smth'
    assert l.list_pins()[1].name == 'p2'
    assert l.get_coeff('p1', 'p1', 'p2', l.prms, tc.tensor(0.0, device=pu)) == tc.tensor(1e4, device=pu)
    print(l.get_coeff('p1', 'p1', 'p2', l.prms, tc.tensor(2, device=pu)))
    assert round(float(l.get_coeff('p1', 'p1', 'p2', l.prms, tc.tensor(2, device=pu)).abs()), 2) == 2.5
    assert round(float(l.get_coeff('p1', 'p1', 'p2', l.prms, tc.tensor(2, device=pu)).angle()), 2) == -1.57

    # Capacitor test
    c = Capacitor(0.03)
    assert c.type == 'c'
    assert c.prms['c'] == 0.03
    assert c.name is None
    assert c.p1.name is 'p1'
    assert c.p2.prnt_comp is None
    assert c.p2.lims is None
    c = Capacitor(0.06, 'else')
    assert c.name == 'else'
    assert c.p1.prnt_comp == 'else'
    assert c.list_pins()[1].name == 'p2'
    assert tc.eq(c.get_coeff('p1', 'p1', 'p2', c.prms, tc.tensor(2, device=pu)),
                 tc.complex(tc.tensor(0.0, device=pu), tc.tensor(0.12, device=pu)))

    # Op amp test
    o = OpAmp(300, 2, 500)
    assert o.type == 'op'
    assert o.prms['rout'] == 2
    assert o.name is None
    assert o.ip.name is 'ip'
    assert o.im.prnt_comp is None
    assert o.lims is None
    assert o.o.lims is None
    o = OpAmp(200, 5, 400, [-2.0, 4.0], 'diff')
    assert o.name == 'diff'
    assert o.im.prnt_comp == 'diff'
    assert o.o.lims[1] == 4.0
    assert o.list_pins()[1].name == 'im'
    assert o.get_coeff('ip', 'im', 'ip', o.prms, 0) == 0.005
    assert o.get_coeff('ip', 'ip', 'o', o.prms, 0) == 0.0
    assert o.get_coeff('o', 'ip', 'o', o.prms, 0) == 80
    assert o.get_coeff('o', 'o', 'im', o.prms, 0) == 0.1
    assert o.get_coeff('o', 'im', 'o', o.prms, 0) == -80
    with pytest.raises(Exception):
        o.get_coeff('o', 'ip', 'im', o.prms, 0)
