"""Hard-coded circuit and experiment models for pre-spice integration BOED."""

import torch as tc
from torch.distributions import constraints
import pyro
import pyro.distributions as dist

R1 = tc.tensor(4, dtype=tc.float)
R2 = tc.tensor(8, dtype=tc.float)


def defected_res_net(v_in):
    # Define the latent variables that determine possible faults and component values, but NOT circuit topology
    r_1 = pyro.sample("R1", dist.Normal(R1, 100.)) # 5% resistor value tolerances
    r_2 = pyro.sample("R2", dist.Normal(R2, 200.))
    r_1_shorted = pyro.sample("R1-S", dist.Bernoulli(0.2))
    r_1_open = pyro.sample("R1-O", dist.Bernoulli(0.2))
    #r_2_shorted = pyro.sample("R2-S", dist.Bernoulli(0.1))
    #r_2_open = pyro.sample("R2-O", dist.Bernoulli(0.1))

    # Define the link functions that determine v_o based on the components and any occurring faults
    v_o = v_in * (r_2 / (r_1 + r_2))
    if r_1_open:
        v_o = v_in * 0.
    elif r_1_shorted:
        v_o = v_in
    return pyro.sample("VO", dist.Normal(v_o, 0.001).to_event(1))


def res_net_guide(v_in):
    r1_mu = pyro.param('R1-mu', tc.tensor(4000.))
    r1_sig = pyro.param('R1-sig', tc.tensor(1000.))
    r2_mu = pyro.param('R2-mu', tc.tensor(8000.))
    r2_sig = pyro.param('R2-sig', tc.tensor(1000.))
    pyro.sample('R1', dist.Normal(r1_mu, r1_sig))
    pyro.sample('R2', dist.Normal(r2_mu, r2_sig))
    r1_s_prior = pyro.param('R1-S-prob', tc.tensor(0.5), constraint=constraints.interval(0., 1.))
    r1_o_prior = pyro.param('R1-O-prob', tc.tensor(0.5), constraint=constraints.interval(0., 1.))
    pyro.sample('R1-S', dist.Bernoulli(r1_s_prior))
    pyro.sample('R1-O', dist.Bernoulli(r1_o_prior))


def res_meas_circuit(v_in, v_o_obs):
    # Define the latent variables that determine possible faults and component values, but NOT circuit topology
    r_1 = pyro.sample('R1', dist.Normal(R1, 0.2)) # 5% resistor value tolerances
    r_2 = pyro.sample('R2', dist.Normal(R2, 0.4))

    # Define the link functions that determine v_o based on the components and any occurring faults
    v_o = v_in * (r_2 / (r_1 + r_2))
    return pyro.sample('VO', dist.Normal(v_o, 0.001).to_event(1), obs=v_o_obs)


def res_meas_guide(v_in, v_o_obs):
    r1_mu = pyro.param('R1-mu', tc.tensor(4.), constraint=constraints.positive)
    r1_sig = pyro.param('R1-sig', tc.tensor(1.), constraint=constraints.positive)
    r2_mu = pyro.param('R2-mu', tc.tensor(8.), constraint=constraints.positive)
    r2_sig = pyro.param('R2-sig', tc.tensor(2.), constraint=constraints.positive)
    pyro.sample('R1', dist.Normal(r1_mu, r1_sig))
    pyro.sample('R2', dist.Normal(r2_mu, r2_sig))


def res_meas_boed(design):
    with pyro.plate_stack("plate", design.shape):
        # Define the latent variables that determine possible faults and component values, but NOT circuit topology
        r_1 = pyro.sample('R1', dist.Normal(R1, 0.2)) # 5% resistor value tolerances
        r_2 = pyro.sample('R2', dist.Normal(R2, 0.4))

        # Define the link functions that determine v_o based on the components and any occurring faults
        v_o = design * (r_2 / (r_1 + r_2))
        out = pyro.sample('VO', dist.Normal(v_o, 0.001))
        return out


def test_guide(design, observation_labels, target_labels):
    vo_mu = pyro.param('VO-mu', tc.tensor(2.))
    vo_sig = pyro.param('VO-sig', tc.tensor(2.))
    pyro.sample('VO', dist.Normal(vo_mu, vo_sig))


def get_fault_val(fault_state, input_expr):
    input_expr = tc.where(fault_state == 2, tc.tensor(1.), input_expr)
    input_expr = tc.where(fault_state == 1, tc.tensor(0.), input_expr)
    return input_expr

def make_mdl(beliefs):
    def simple_dig_faults(design):
        """Fault model for a simple digital circuit defined by o = (i1 or i2) and i3"""
        with pyro.plate_stack('dig-plate', design.shape[:-1]):
            # Define the fault nodes
            n1_f = pyro.sample('N1-F', dist.Categorical(beliefs[0, :]))
            n2_f = pyro.sample('N2-F', dist.Categorical(beliefs[1, :]))
            n3_f = pyro.sample('N3-F', dist.Categorical(beliefs[2, :]))
            n4_f = pyro.sample('N4-F', dist.Categorical(beliefs[3, :]))
            n5_f = pyro.sample('N5-F', dist.Categorical(beliefs[4, :]))

            # Now define the true values for each node
            n1 = get_fault_val(n1_f, design[..., 0])
            n2 = get_fault_val(n2_f, design[..., 1])
            n3 = get_fault_val(n3_f, design[..., 2])
            n4 = get_fault_val(n4_f, tc.logical_or(n1, n2, out=tc.empty(n1.shape, dtype=tc.float)))
            n5 = get_fault_val(n5_f, tc.logical_and(n4, n3, out=tc.empty(n4.shape, dtype=tc.float)))
            return pyro.sample('O', dist.Normal(n5, 0.001))
    return simple_dig_faults


def dig_guide(design):
    n1c = pyro.param('N1-C', tc.tensor(0.34), constraint=constraints.positive)
    n10 = pyro.param('N1-0', tc.tensor(0.33), constraint=constraints.positive)
    n11 = pyro.param('N1-1', tc.tensor(0.33), constraint=constraints.positive)
    pyro.sample('N1-F', dist.Categorical(tc.tensor([n1c, n10, n11])))
    n2c = pyro.param('N2-C', tc.tensor(0.34), constraint=constraints.positive)
    n20 = pyro.param('N2-0', tc.tensor(0.33), constraint=constraints.positive)
    n21 = pyro.param('N2-1', tc.tensor(0.33), constraint=constraints.positive)
    pyro.sample('N2-F', dist.Categorical(tc.tensor([n2c, n20, n21])))
    n3c = pyro.param('N3-C', tc.tensor(0.34), constraint=constraints.positive)
    n30 = pyro.param('N3-0', tc.tensor(0.33), constraint=constraints.positive)
    n31 = pyro.param('N3-1', tc.tensor(0.33), constraint=constraints.positive)
    pyro.sample('N3-F', dist.Categorical(tc.tensor([n3c, n30, n31])))
    n4c = pyro.param('N4-C', tc.tensor(0.34), constraint=constraints.positive)
    n40 = pyro.param('N4-0', tc.tensor(0.33), constraint=constraints.positive)
    n41 = pyro.param('N4-1', tc.tensor(0.33), constraint=constraints.positive)
    pyro.sample('N4-F', dist.Categorical(tc.tensor([n4c, n40, n41])))
    n5c = pyro.param('N5-C', tc.tensor(0.34), constraint=constraints.positive)
    n50 = pyro.param('N5-0', tc.tensor(0.33), constraint=constraints.positive)
    n51 = pyro.param('N5-1', tc.tensor(0.33), constraint=constraints.positive)
    pyro.sample('N5-F', dist.Categorical(tc.tensor([n5c, n50, n51])))


def dig_boed_guide(design, obs_lbls, trgt_lbls):
    vo_mu = pyro.param('O-mu', tc.tensor(0.5))
    vo_sig = pyro.param('O-sig', tc.tensor(1.))
    pyro.sample('O', dist.Normal(vo_mu, vo_sig))

