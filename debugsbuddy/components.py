import torch as tc

__all__ = ['Pin', 'Component', 'Resistor', 'Inductor', 'Capacitor', 'OpAmp', 'VoltSource', 'Ground']

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
zero = tc.tensor(0.0, device=pu)


class Pin:
    def __init__(self, name: str, parent_component: str, v_lims: list[float] = None):
        self.name = name
        self.prnt_comp = parent_component
        self.lims = v_lims


class Component:
    pass


class Resistor(Component):
    def __init__(self, r, name: str = None):
        self.type = 'r'
        self.prms = {'r': r}
        self.name = name
        self.p1 = Pin('p1', name)
        self.p2 = Pin('p2', name)

    def list_pins(self):
        return [self.p1, self.p2]

    @staticmethod
    def get_coeff(node, p1, p2, prms, freq):
        return 1 / prms['r']


class Inductor(Component):
    def __init__(self, l, name: str = None):
        self.type = 'l'
        self.prms = {'l': l}
        self.name = name
        self.p1 = Pin('p1', name)
        self.p2 = Pin('p2', name)

    def list_pins(self):
        return [self.p1, self.p2]

    @staticmethod
    def get_coeff(node, p1, p2, prms, freq):
        return tc.where(freq == 0.0, tc.tensor(1e4, device=pu), 1 / tc.complex(zero, freq * prms['l']))


class Capacitor(Component):
    def __init__(self, c, name: str = None):
        self.type = 'c'
        self.prms = {'c': c}
        self.name = name
        self.p1 = Pin('p1', name)
        self.p2 = Pin('p2', name)

    def list_pins(self):
        return [self.p1, self.p2]

    @staticmethod
    def get_coeff(node, p1, p2, prms, freq):
        return tc.complex(zero, freq * prms['c'])


class VoltSource(Component):
    def __init__(self, min: float = None, max: float = None, name: str = None):
        self.type = 'vin'
        self.range = (min, max)
        self.name = name
        self.p1 = Pin('vin', name)

    def list_pins(self):
        return [self.p1]

    @staticmethod
    def get_coeff(node, p1, p2, prms, freq):
        raise Exception('No pin relations for component with only one pin')


class Ground(Component):
    def __init__(self, name: str = None):
        self.type = 'gnd'
        self.name = name
        self.p1 = Pin('gnd', name)

    def list_pins(self):
        return [self.p1]

    @staticmethod
    def get_coeff(node, p1, p2, prms, freq):
        raise Exception('No pin relations for component with only one pin')


class OpAmp(Component):
    def __init__(self, rin, rout, gain, rails: list[float] = None, name: str = None):
        self.type = 'op'
        self.prms = {'rin': rin, 'rout': rout, 'gain': gain}
        self.lims = rails
        self.name = name
        self.ip = Pin('ip', name)
        self.im = Pin('im', name)
        self.o = Pin('o', name, rails)

    def list_pins(self):
        return [self.ip, self.im, self.o]

    @staticmethod
    def get_coeff(node, p1, p2, prms, freq):
        # The opamp coefficients depend on which node equation we're constructing, and the order of the pin pair
        if node in ['ip', 'im']:
            if p1 in ['ip', 'im'] and p2 in ['ip', 'im']:
                return 1 / (100 * prms['rin'])
            else:
                return zero
        elif node == 'o':
            if p1 == 'o' and p2 in ['ip', 'im']:
                # This coefficient is half the correct value as it will be added twice, once for each input
                return 1 / (2 * prms['rout'])
            elif p1 == 'ip' and p2 == 'o':
                return (100 * prms['gain']) / prms['rout']
            elif p1 == 'im' and p2 == 'o':
                return -((100 * prms['gain']) / prms['rout'])
        # If we didn't return yet, that means in invalid set of arguments was provided
        raise Exception('Invalid opamp pin pair relationship requested')
