"""Entry point for the BOED-based analog circuit debug tool."""

import time
import itertools
from scipy.stats import norm
import torch as tc
import pyro
from pyro import poutine
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

__all__ = ['guided_debug']

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
print(f"Using processor {pu}")

# Defaults for debugs buddy tuning parameters
EIG_SAMPLES = 6000
INF_SAMPLES = 1e6
BLAME_THRESHOLD = 0.1
DISCRETE_VOLTS = 11
DISCRETE_FREQS = 13

OPEN_ADMITTANCE = 1e-2
SHRT_ADMITTANCE = 1e2
OPEN_ADMITTANCE_INF = 1e-4
SHRT_ADMITTANCE_INF = 1e4
OPEN_FAULT_PROB = 0.1
SHRT_FAULT_PROB = 0.05
COMP_PRM_SPREAD = 0.2
COMP_PRM_TOL = 0.1
MEAS_ERROR = 0.002


def eval_eigs(prob_mdl, tests, obs_labels, circ_prm_labels, eig_samples):
    """
    Compute (estimate) the expected information gain of each candidate test given the current probabilistic circuit
    model state.
    """
    eig = nmc_eig(
        prob_mdl,
        tests,           # design, or in this case, tensor of possible designs
        obs_labels,      # site label of observations, could be a list
        circ_prm_labels, # site label of 'targets' (latent variables), could also be list
        N=eig_samples,           # number of theta samples to use
        M=eig_samples)           # number of observation samples
    accepted_eigs = eig.cpu()
    return accepted_eigs.detach()


def condition_fault_model(fault_mdl, inputs, measured, latent_lbls, inf_samples):
    """
    Numerically estimate the posterior probabilities of the various candidate faults within a circuit using Bayes'
    rule conditioned on the observed circuit measurements.

    Currently, we use importance sampling for the numerical estimation technique.
    """

    # First generate a bunch of samples from the posterior and accumulate the log probability of each output sample
    cond_mdl = pyro.condition(fault_mdl, measured)
    n_ins = pyro.contrib.util.lexpand(inputs, int(inf_samples))
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
    resamples = tc.distributions.Categorical(normed_w).sample((int(inf_samples),))
    print(f"Number of samples used to construct updated beliefs: {len(tc.unique(resamples))}")

    # Now take the latent values from each trace in the resampled set and average to get the updated set of beliefs
    new_blfs = {}
    for ltnt in latent_lbls:
        # Handle edges first
        vals = trace.nodes[ltnt]['value']
        sampled = tc.take(vals, resamples)
        if ltnt[0] == 'e':
            new_blfs[ltnt] = tc.count_nonzero(sampled == 1) / sampled.size(0)
        else:
            mu, std = norm.fit(sampled.cpu().numpy())
            comp, prm = ltnt.split('-')
            if comp in new_blfs:
                new_blfs[comp][prm] = tc.tensor([mu, std], device=pu)
            else:
                new_blfs[comp] = {prm: tc.tensor([mu, std], device=pu)}
    return new_blfs


def analyze_beliefs(init: dict, new: dict, blame_thrshld: float):
    worst_edges = [{'ltnt': 'none', 'diff': 0}, {'ltnt': 'none', 'diff': 0}, {'ltnt': 'none', 'diff': 0}]
    worst_param = {'ltnt': 'none', 'stddevs': 0}
    worst_uncertainty = ''
    likely_correct = True
    bad_state = lambda x: 'unconnected' if x > 0.5 else 'shorted'
    problems = []

    for ltnt in new:
        # Latent variables that are beliefs about whether a node connection is short vs. open
        if ltnt[0] == 'e':
            # We take the difference opposite ways for connections that are intended to be short vs. open
            # This way a positive difference always indicates that the connection is now more likely to be correct
            d = init[ltnt] - new[ltnt] if init[ltnt] < 0.5 else new[ltnt] - init[ltnt]
            # The circuit is only anticipated to be correct if all edges have tended towards the correct states
            if d < -blame_thrshld:
                likely_correct = False
                # Determine if the change in belief is bad enough to place it correctly in the worst three edges
                if d < worst_edges[2]['diff']:
                    if d < worst_edges[1]['diff']:
                        if d < worst_edges[0]['diff']:
                            worst_edges[0] = {'ltnt': ltnt, 'diff': d, 'problem': bad_state(init[ltnt])}
                        else:
                            worst_edges[1] = {'ltnt': ltnt, 'diff': d, 'problem': bad_state(init[ltnt])}
                    else:
                        worst_edges[2] = {'ltnt': ltnt, 'diff': d, 'problem': bad_state(init[ltnt])}
        else:
            # For component parameters we determine the difference in both normal distribution parameters
            for prm in new[ltnt]:
                tol_diff = init[ltnt][prm][0] * COMP_PRM_TOL
                mismatch = new[ltnt][prm][0] + tol_diff < init[ltnt][prm][0] or new[ltnt][prm][0] - tol_diff > init[ltnt][prm][0]
                if mismatch:
                    problems.append({'ltnt': f"{ltnt}-{prm}", 'expected': init[ltnt][prm][0], 'inferred': new[ltnt][prm][0]})

    # Construct the list of potential circuit construction problems to report to the user
    if not likely_correct:
        if worst_uncertainty != '':
            print(f"Less certain about the value of {worst_uncertainty} than at the start of debugging, unusual...")
        # Don't want to overload the user by suggesting four possible problems; try to reduce to the most likely problem
        if worst_edges[0]['diff'] < 0:
            # If the worst edge is more than twice as bad as the next worst don't even bother reporting the other ones,
            # and similarly if the 2nd worst is more than twice as bad as the 3rd
            if (2 * worst_edges[1]['diff']) <= worst_edges[0]['diff']:
                if (2 * worst_edges[2]['diff']) <= worst_edges[1]['diff']:
                    problems.append(worst_edges[2])
                problems.append(worst_edges[1])
            problems.append(worst_edges[0])
        # Add the worst parameter if it exists since there's no way to effectively compare the 'badness' to the edges
        if worst_param['stddevs'] < 0:
            problems.append(worst_param)
    return problems


def guided_debug(circuit, mode='simulated', single_iter=False, viz_eigs=False, **prm_overrides):
    # Setup of general objects needed for the guided debug process
    print(f"Starting guided debug using Debugs Buddy...")

    # Handle overrides of defaults for method parameters (analogous to hyper-parameters)
    eig_samples = prm_overrides['eig_samples'] if 'eig_samples' in prm_overrides else EIG_SAMPLES
    inf_samples = prm_overrides['inf_samples'] if 'inf_samples' in prm_overrides else INF_SAMPLES
    blame_thrshld = prm_overrides['blame_threshold'] if 'blame_threshold' in prm_overrides else BLAME_THRESHOLD
    volt_steps = prm_overrides['discrete_volt_steps'] if 'discrete_volt_steps' in prm_overrides else DISCRETE_VOLTS
    freq_steps = prm_overrides['discrete_freq_steps'] if 'discrete_freq_steps' in prm_overrides else DISCRETE_FREQS

    shrt = tc.tensor(prm_overrides['shrt_admittance'] if 'shrt_admittance' in prm_overrides else SHRT_ADMITTANCE, device=pu)
    open = tc.tensor(prm_overrides['open_admittance'] if 'open_admittance' in prm_overrides else OPEN_ADMITTANCE, device=pu)
    shrt_inf = tc.tensor(prm_overrides['shrt_admit_inf'] if 'shrt_admit_inf' in prm_overrides else SHRT_ADMITTANCE_INF, device=pu)
    open_inf = tc.tensor(prm_overrides['open_admit_inf'] if 'open_admit_inf' in prm_overrides else OPEN_ADMITTANCE_INF, device=pu)
    shrt_prob = prm_overrides['shrt_fault_prob'] if 'shrt_fault_prob' in prm_overrides else SHRT_FAULT_PROB
    open_prob = prm_overrides['open_fault_prob'] if 'open_fault_prob' in prm_overrides else OPEN_FAULT_PROB
    prm_spread = prm_overrides['comp_prm_spread'] if 'comp_prm_spread' in prm_overrides else COMP_PRM_SPREAD
    meas_error = prm_overrides['meas_error'] if 'meas_error' in prm_overrides else MEAS_ERROR

    # Set up the compute device
    tc.backends.cuda.matmul.allow_tf32 = True
    # Determine if circuit requires AC analysis
    circ_is_ac = False
    for comp in circuit.comps.values():
        if comp.type in ['c', 'l']:
            circ_is_ac = True

    # Define the initial fault model and the graphical nodes that we will be conditioning and observing
    curr_mdl = circuit.gen_fault_mdl(shrt_res=shrt, open_res=open, shrt_fault_prob=shrt_prob, open_fault_prob=open_prob,
                                     comp_prm_spread=prm_spread, meas_error=meas_error, complex=circ_is_ac)
    curr_inf_mdl = circuit.gen_fault_mdl(shrt_res=shrt_inf, open_res=open_inf, shrt_fault_prob=shrt_prob, open_fault_prob=open_prob,
                                         comp_prm_spread=prm_spread, meas_error=meas_error, complex=circ_is_ac)
    obs_lbls = circuit.get_obsrvd_lbls(circ_is_ac)
    ltnt_lbls = circuit.get_latent_lbls()
    init_beliefs = circuit.curr_beliefs

    # Construct test voltages
    volts = {}
    for comp in circuit.comps.values():
        # For each input voltage, we consider possible amplitudes in increments of 100mV within the voltage source range
        if comp.type == 'vin':
            volts[comp.name] = tc.linspace(comp.range[0], comp.range[1], volt_steps, dtype=tc.float, device=pu)

    if circ_is_ac:
        # Construct frequencies
        freqs = tc.logspace(-2, 2, freq_steps, dtype=tc.float, device=pu) if circ_is_ac else tc.tensor([0.0])
        # Put the voltages and frequencies together into the complete BOED input matrix
        candidate_tests = tc.tensor(list(itertools.product(*volts.values(), freqs)), dtype=tc.float, device=pu)
    else:
        candidate_tests = tc.tensor(list(itertools.product(*volts.values())), dtype=tc.float, device=pu)

    # With the circuit and experiments defined, begin recommending measurements to determine implementation faults
    pyro.clear_param_store()
    while True:
        start = time.time()
        # First we determine what test inputs to apply to the circuit next
        print(f"Determining next best test to conduct...")
        eigs = None
        # We use batching to reduce the problem size so that each subproblem can fit on the GPU
        candidate_tests = tc.split(candidate_tests, 3)
        for batch in candidate_tests:
            if eigs is None:
                eigs = eval_eigs(curr_mdl, batch, obs_lbls, ltnt_lbls, eig_samples)
            else:
                eigs2 = eval_eigs(curr_mdl, batch, obs_lbls, ltnt_lbls, eig_samples)
                eigs = tc.concat((eigs, eigs2))
        best_test = int(tc.argmax(eigs).detach())
        candidate_tests = tc.concat(candidate_tests)

        viz_results = True
        if viz_results:
            plt.figure(figsize=(20, 7))
            # TODO: Make x_vals determination more general
            if circ_is_ac:
                x_vals = [f"{round(float(test[0]), 1)}, {int(test[1])}" for test in candidate_tests]
            else:
                x_vals = [f"{round(float(test[0]), 1)}, {round(float(test[1]), 1)}" for test in candidate_tests]
            plt.plot(x_vals, eigs.numpy(), marker='o', linewidth=2)
            plt.xlabel("Possible inputs")
            plt.xticks(rotation=90, fontsize='x-small')
            plt.ylabel("EIG")
            plt.show()

        alt_viz = False
        if alt_viz:
            p, ax = plt.subplots(figsize=(20, 7))
            #df = pd.DataFrame({'eig': eigs.cpu(), 'v': candidate_tests[:, 0].cpu(), 'f': candidate_tests[:, 1].cpu()})
            #seaborn.scatterplot(df, x='v', y='f', palette='viridis', hue='eig', hue_norm=(0, 10))
            ax.contourf(freqs.cpu(), volts['vin1'].cpu(), eigs.cpu().reshape((len(volts['vin1']), len(freqs))))
            ax.set_xscale('log')
            plt.show()

        # Apply the selected test inputs to the circuit and collect measurements
        if mode == 'simulated':
            print(f"Next best test: {candidate_tests[best_test]}.")
            measured = circuit.simulate_actual(candidate_tests[best_test], circ_is_ac).to(pu)
            print(f"Measured from test: {measured}.")
        else:
            # If in real-world guided debug mode the user must collect the measurements manually
            print(f"Next best test: {candidate_tests[best_test]}.")
            print(f"Please apply these values and collect measurements at {str(obs_lbls)}.")
            measured = input('Input measurements as floats separated by spaces...')
            measured = [tc.tensor(float(val), device=pu) for val in measured.split(' ')]

        # Now we condition the fault model on the measured data
        obs_set = {}
        for i, obs in enumerate(obs_lbls):
            obs_set[obs] = measured[i]
        print(f"Updating probable faults based on measurement data...")
        new_beliefs = condition_fault_model(curr_inf_mdl, candidate_tests[best_test], obs_set, ltnt_lbls, inf_samples)

        # Analyze the new beliefs to try and determine whether there's a fault and what it is likely to be
        problems = analyze_beliefs(init_beliefs, new_beliefs, blame_thrshld)
        # The circuit is likely to be correct if all beliefs became more certain
        p_list = []
        if len(problems) == 0:
            print("Measurements indicate that the circuit is likely to be correctly assembled!")
        else:
            print("Problems likely exist, try checking the following:")
            for p in problems:
                if p['ltnt'][0] == 'e':
                    # If the problem is an edge, let the user know which two nodes it links
                    n1, n2 = p['ltnt'][2:].split('-')
                    n1 = circuit.nodal_name_from_index(int(n1))
                    n2 = circuit.nodal_name_from_index(int(n2))
                    print(f"   -Looks like nodes {n1} and {n2} might be {p['problem']}")
                    p_list.append((n1, n2))
                else:
                    print(f"   -Looks like component parameter {p['ltnt']} might be {p['inferred']} instead of {p['expected']}")
                    p_list.append(p['ltnt'])

        # Update the fault model to reflect the new beliefs and run another iteration
        curr_mdl = circuit.gen_fault_mdl(new_beliefs, shrt_res=shrt, open_res=open,
                                         shrt_fault_prob=shrt_prob, open_fault_prob=open_prob,
                                         comp_prm_spread=prm_spread, meas_error=meas_error, complex=circ_is_ac)
        curr_inf_mdl = circuit.gen_fault_mdl(new_beliefs, shrt_res=shrt_inf, open_res=open_inf,
                                             shrt_fault_prob=shrt_prob, open_fault_prob=open_prob,
                                             comp_prm_spread=prm_spread, meas_error=meas_error, complex=circ_is_ac)
        print(new_beliefs)
        print(f'Iter time taken: {time.time() - start}s')
        if single_iter:
            return p_list
        else:
            input('Press Enter to run another iteration...')
