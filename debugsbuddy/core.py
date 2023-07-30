"""Entry point for the BOED-based analog circuit debug tool."""

import itertools

import time
import multiprocessing as mp

import numpy as np
from scipy.stats import norm
import torch as tc
import pyro
from pyro import poutine
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
import matplotlib.pyplot as plt

__all__ = ['guided_debug']

example_circuit = None

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
print(pu)


def eval_eigs(prob_mdl, tests, obs_labels=None, circ_prm_labels=None, viz_results=False, pu=None):
    """
    Compute (estimate) the expected information gain of each candidate test given the current probabilistic circuit
    model state.
    """
    eig = nmc_eig(
        prob_mdl,
        tests,           # design, or in this case, tensor of possible designs
        obs_labels,      # site label of observations, could be a list
        circ_prm_labels, # site label of 'targets' (latent variables), could also be list
        N=8000,           # number of samples to draw per step in the expectation
        M=8000)           # number of gradient steps
    accepted_eigs = eig.cpu()
    return accepted_eigs.detach()


def condition_fault_model(fault_mdl, inputs, measured, prms, edges, old_beliefs):
    """
    Numerically estimate the posterior probabilities of the various candidate faults within a circuit using Bayes'
    rule conditioned on the observed circuit measurements.

    Currently, we use importance sampling for the numerical estimation technique.
    """

    # First generate a bunch of samples from the posterior and accumulate the log probability of each output sample
    cond_mdl = pyro.condition(fault_mdl, measured)
    n_ins = pyro.contrib.util.lexpand(inputs, int(1e6))
    trace = poutine.trace(cond_mdl).get_trace(n_ins)
    trace.compute_log_prob()
    # Accumulate the log probability weights 'w' for each sample
    log_ws = None
    for node in measured:
        if log_ws is None:
            log_ws = trace.nodes[node]['log_prob']
        else:
            log_ws += trace.nodes[node]['log_prob']
    # Normalize the weights so that the sum across all samples is 1
    log_w_norm = log_ws - tc.logsumexp(log_ws, 0)
    normed_w = tc.exp(log_w_norm)

    # Now sample from the set of sampled outputs based on the log probabilities, the resample values are trace indices
    resamples = tc.distributions.Categorical(normed_w).sample((int(1e6),))
    print(f"Number of samples used to construct updated beliefs: {len(tc.unique(resamples))}")

    # Now take the latent values from each trace in the resampled set and average to get the updated set of beliefs
    new_blfs = {}
    for prm in prms:
        #sampled = tc.tensor([trace.nodes[prm]['value'][s] for s in resamples])
        #mu, std = norm.fit(sampled)
        #new_blfs[prm] = tc.tensor([mu, std], device=pu)
        new_blfs[prm] = old_beliefs[prm]
    for edge in edges:
        edge_name = str(sorted(tuple(edge)))
        vals = trace.nodes[edge_name]['value']
        sampled = tc.take(vals, resamples)
        new_blfs[edge_name] = tc.count_nonzero(sampled == 1, ) / sampled.size(0)
    return new_blfs


def guided_debug(circuit=example_circuit, mode='simulated'):
    # Setup of general objects needed for the guided debug process
    print(f"Starting guided debug using Debugs Buddy...")

    # Setup compute device
    tc.backends.cuda.matmul.allow_tf32 = True

    # Construct possible test voltages to apply
    #v_list = tc.linspace(0, 1, 11, device=pu)
    ## With two test voltages we consider every 100mV steps and every possible combination of the two sources
    #candidate_tests = tc.tensor(list(product(v_list, repeat=2)), dtype=tc.float, device=pu)
    #gnds = tc.zeros(121, device=pu).unsqueeze(-1)
    #vccs = tc.ones(121, device=pu).unsqueeze(-1)
    #candidate_tests = tc.cat((candidate_tests, gnds), -1)
    #if vcc:
    #    candidate_tests = tc.cat((candidate_tests, vccs), -1)
    #    reduced_tests = []
    #    for test in candidate_tests:
    #        if (test[0] <= test[1]) and ((test[1] - test[0]) <= 0.3):
    #            reduced_tests.append(test)
    #    candidate_tests = tc.stack(reduced_tests)

    # Construct test voltages
    volts = {}
    for comp in circuit.comps.values():
        # For each input voltage, we consider possible amplitudes in increments of 100mV within the voltage source range
        if comp.type == 'vin':
            num_steps = int((comp.range[1] - comp.range[0]) / 0.1) + 1
            volts[comp.name] = tc.linspace(comp.range[0], comp.range[1], num_steps, dtype=tc.float, device=pu)
    # Construct frequencies
    # For now we just consider 1Hz (10^0) to 1MHz (10^6) in log steps
    freqs = tc.logspace(0, 6, 13, dtype=tc.float, device=pu)
    # Put the voltages and frequencies together into the complete BOED input matrix
    candidate_tests = tc.tensor(list(itertools.product(*volts.values(), freqs)), dtype=tc.float, device=pu)
    #candidate_tests = tc.stack(candidate_inputs)

    # Define the initial fault model and the graphical nodes that we will be conditioning and observing
    curr_mdl = circuit.gen_fault_mdl()
    obs_lbls = circuit.get_obsrvd_lbls()
    ltnt_lbls = circuit.get_latent_lbls()

    # With the circuit to debug defined, we can begin recommending measurements to determine implementation faults
    pyro.clear_param_store()
    while True:
        # First we determine what test inputs to apply to the circuit next
        print(f"Determining next best test to conduct...")
        eigs = None
        # We use batching to reduce the problem size so that each subproblem can fit on the GPU
        candidate_tests = tc.split(candidate_tests, 4)
        for batch in candidate_tests:
            if eigs is None:
                eigs = eval_eigs(curr_mdl, batch, obs_lbls, ltnt_lbls)
            else:
                eigs2 = eval_eigs(curr_mdl, batch, obs_lbls, ltnt_lbls)
                eigs = tc.concat((eigs, eigs2))
        best_test = int(tc.argmax(eigs).detach())
        candidate_tests = tc.concat(candidate_tests)

        viz_results = True
        if viz_results:
            plt.figure(figsize=(20, 7))
            x_vals = [f"{round(float(test[0]), 1)}, {int(test[1])}" for test in candidate_tests]
            plt.plot(x_vals, eigs.numpy(), marker='o', linewidth=2)
            plt.xlabel("Possible inputs")
            plt.xticks(rotation=90, fontsize='x-small')
            plt.ylabel("EIG")
            plt.show()

        # Apply the selected test inputs to the circuit and collect measurements
        if mode == 'simulated':
            print(f"Next best test: {candidate_tests[best_test]}.")
            measured = circuit.simulate_test(candidate_tests[best_test]).to(pu)
            print(f"Measured from test: {measured}.")
        else:
            # If in real-world guided debug mode the user must collect the measurements manually
            print(f"Next best test: {candidate_tests[best_test]}.")
            print(f"Please apply these values and collect measurements at {str(obs_lbls)}.")
            measured = input('Input measurements as floats separated by spaces...')
            measured = [tc.tensor(float(val), device=pu) for val in measured.split(' ')]

        obs_set = {}
        j = 0
        for i, node in enumerate(circuit.nodes):
            if node.name in obs_lbls:
                obs_set[node.name] = measured[j]
                j += 1

        # Now we condition the fault model on the measured data
        print(f"Updating probable faults based on measurement data...")
        new_beliefs = condition_fault_model(curr_mdl, candidate_tests[best_test], obs_set,
                                            circuit.comp_prms, circuit.edges, circuit.priors)
        curr_mdl = circuit.gen_fault_mdl(new_beliefs)

        # Now print the probable circuit model for the user to view
        print('Correct:')
        print(circuit.correct)
        print('Beliefs updated:')
        print(new_beliefs)

        # Try to analyze the output to identify faulty construction
        correct_count, total_edges = 0, 0
        for prior in circuit.priors:
            if new_beliefs[prior].dim() == 0:
                total_edges += 1
                diff = tc.abs(new_beliefs[prior] - circuit.correct[prior])
                if diff > 0.3:
                    print(f"Looks like edge {prior} is either shorted or unconnected erroneously!\n"
                          f"Belief: {new_beliefs[prior]}, correct: {circuit.correct[prior]}")
                elif diff < 0.1:
                    correct_count += 1
        print(f"Currently {correct_count} edges out of {total_edges} are anticipated to be correct.")
        if total_edges == correct_count:
            print(f"Seems like your circuit is constructed correctly!")

        input('Press Enter to run another cycle...')


if __name__ == '__main__':
    guided_debug()

