"""Entry point for the BOED-based analog circuit debug tool."""

import numpy as np
import torch as tc
import pyro
from pyro.contrib.oed.eig import nmc_eig
import pyro.distributions as dist
import matplotlib.pyplot as plt


class CUT:
    def __init__(self, n1, n2, n3, n4, n5):
        self.n1_f = n1
        self.n2_f = n2
        self.n3_f = n3
        self.n4_f = n4
        self.n5_f = n5

    def run(self, i1, i2, i3):
        n1 = self.n1_f if self.n1_f is not None else i1
        n2 = self.n2_f if self.n2_f is not None else i2
        n3 = self.n3_f if self.n3_f is not None else i3
        n4 = self.n4_f if self.n4_f is not None else int(n1 or n2)
        n5 = self.n5_f if self.n5_f is not None else int(n4 and n3)
        return n5


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
            return pyro.sample('O', dist.Normal(n5, 0.1))
    return simple_dig_faults


def eval_test_eigs(mdl, designs, viz_results: bool = False):
    # Now for a BOED phase
    eig = nmc_eig(
        mdl,
        designs,       # design, or in this case, tensor of possible designs
        ['O'],                  # site label of observations, could be a list
        ['N1-F', 'N2-F', 'N3-F', 'N4-F', 'N5-F'],      # site label of 'targets' (latent variables), could also be list
        N=50000,         # number of samples to draw per step in the expectation
        M=500)

    if viz_results:
        plt.figure(figsize=(10,5))
        x_vals = ['000', '001', '010', '011', '100', '101', '110', '111']
        plt.plot(x_vals, eig.detach().numpy(), marker='o', linewidth=2)
        plt.xlabel("Input set")
        plt.ylabel("EIG")
        plt.show()
    return eig.detach()


def amortized_dig_debug():
    # First define the "implemented" circuit
    test_circuit = CUT(None, None, None, None, None)
    # Define the test design space
    candidate_designs = tc.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tc.float)
    for test in candidate_designs:
        print(f"When input = {test}, output = {test_circuit.run(*test)}")

    # Construct our initial beliefs about possible circuit faults
    fault_priors = tc.tensor([0.9, 0.05, 0.05])
    beliefs = fault_priors.tile((5, 1))
    pyro.clear_param_store()

    for test_pattern in range(6):
        print(f"Beginning round {test_pattern}")
        # Construct current model based on beliefs
        curr_mdl = make_mdl(beliefs)

        # First determine best test pattern to apply
        best = int(tc.argmax(eval_test_eigs(curr_mdl, candidate_designs, True)).float().detach())

        # Apply the test pattern to the actual circuit
        print(f"Applying test pattern {candidate_designs[best, :]}")
        out = test_circuit.run(*candidate_designs[best, :])
        print(f"Result: {out}")

        # Condition the model based on the outputs
        cond_mdl = pyro.condition(curr_mdl, {'O': tc.tensor(out)})

        sampler = pyro.infer.Importance(cond_mdl, num_samples=1000)
        results = sampler.run(candidate_designs[best, :])
        normed_w = sampler.get_normalized_weights()
        resamples = tc.distributions.Categorical(normed_w).sample((1000,))
        sampled_vals = {}
        sampled_vals['N1-F'] = [results.exec_traces[s].nodes['N1-F']['value'] for s in resamples]
        sampled_vals['N2-F'] = [results.exec_traces[s].nodes['N2-F']['value'] for s in resamples]

        # Update the current model based on the posterior
        for i in range(5):
            sampled_vals = np.array([results.exec_traces[s].nodes[f"N{i+1}-F"]['value'] for s in resamples])
            beliefs[i, 0] = np.count_nonzero(sampled_vals == 0) / sampled_vals.size
            beliefs[i, 1] = np.count_nonzero(sampled_vals == 1) / sampled_vals.size
            beliefs[i, 2] = np.count_nonzero(sampled_vals == 2) / sampled_vals.size
            print(f"Round {test_pattern} updated beliefs for N{i+1}: {beliefs[i, :]}")


if __name__ == '__main__':
    amortized_dig_debug()
