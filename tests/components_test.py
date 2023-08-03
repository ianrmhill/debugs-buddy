import torch as tc
import pytest

from debugsbuddy.components import *


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
