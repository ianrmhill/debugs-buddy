"""Entry point for the BOED-based analog circuit debug tool."""

import numpy as np
import torch as tc
import pyro
import pyro.distributions as dist
from pyro.contrib.oed.eig import nmc_eig
from pyro.optim import Adam
import matplotlib.pyplot as plt
from itertools import product


class WeighResult:
    def __init__(self, which_ball: int):
        # Which ball is an int between 1 and 8 representing the location of the heavier ball
        self.w_diff = np.ones(8)
        self.w_diff[which_ball-1] = 1.1

    def run(self, positions):
        w_one = np.sum(np.where(positions == 0, 1, 0) * self.w_diff)
        w_two = np.sum(np.where(positions == 1, 1, 0) * self.w_diff)
        if w_one > w_two:
            scale_state = 1
        elif w_one < w_two:
            scale_state = -1
        else:
            scale_state = 0
        return scale_state


def make_mdl(priors):
    def riddle_mdl(design):
        with pyro.plate_stack('weigh-plate', design.shape[:-1]):
            balls = pyro.sample('BH', dist.Categorical(priors))
            pos_one = tc.where(design == 0, 1, 0)
            pos_two = tc.where(design == 1, 1, 0)
            # Wow! I hate tensor indexing
            # These lines sum up all the weights, and add a 0.1 weight if the heavier ball is one of the balls
            w_one = tc.sum(pos_one, dim=-1) + (tc.gather(pos_one, -1, balls.unsqueeze(-1)).squeeze(-1) * 0.1)
            w_two = tc.sum(pos_two, dim=-1) + (tc.gather(pos_two, -1, balls.unsqueeze(-1)).squeeze(-1) * 0.1)

            scale_state = tc.where(w_one > w_two, 1, 0)
            scale_state = tc.where(w_one < w_two, -1, scale_state).float()#.unsqueeze(-1)
            out = pyro.sample('Scale-State', dist.Normal(scale_state, 0.01))
            return out
    return riddle_mdl



def eval_test_eigs(mdl, designs, num_balls, viz_results: bool = False):
    # Now for a BOED phase
    optimizer = Adam({'lr': 0.4})
    eig = nmc_eig(
        mdl,
        designs,       # design, or in this case, tensor of possible designs
        ['Scale-State'],         # site label of observations, could be a list
        ['BH'], # site label of 'targets' (latent variables), could also be list
        N=1000,         # number of samples to draw per step in the expectation
        M=100)     # number of gradient steps

    if viz_results:
        bins = np.zeros(4)
        counts = np.zeros(4)
        for i, val in enumerate(eig.detach().numpy()):
            bins[num_balls[i] - 1] += val
            counts[num_balls[i] - 1] += 1
        bins /= counts

        plt.figure(figsize=(10,5))
        x_vals = ['1', '2', '3', '4']
        plt.plot(x_vals, bins, marker='o', linewidth=2)
        plt.xlabel("Input set")
        plt.ylabel("EIG")
        plt.show()
    return eig.detach()


def riddle_solve():
    scale = WeighResult(6)
    # List out all 6561 possible ways to weigh 8 balls
    all_designs = tc.tensor(list(product([0, 1, 2], repeat=8)), dtype=tc.int)
    # Only keep those where the same number of balls are on each side of the scale
    candidate_designs = []
    balls_per_side = []
    for design in range(all_designs.shape[0]):
        counts = tc.bincount(all_designs[design, :], minlength=3)
        if counts[0] == counts[1]:
            candidate_designs.append(all_designs[design, :])
            balls_per_side.append(counts[0].numpy())
    candidate_designs = tc.stack(candidate_designs)


    # Initially the heavier ball is equally likely to be any of the eight balls
    ball_priors = tc.tensor([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    print(f"Prior probabilities for index of heavier ball: {ball_priors[:]}")
    pyro.clear_param_store()

    for test in range(2):
        print(f"Beginning round {test}")
        # Construct current model based on beliefs
        curr_mdl = make_mdl(ball_priors)

        # First determine best test pattern to apply
        best = int(tc.argmax(eval_test_eigs(curr_mdl, candidate_designs, balls_per_side, True)).float().detach())

        # Apply the test pattern to the actual circuit
        print(f"Applying test pattern {candidate_designs[best]}")
        out = scale.run(candidate_designs[best])
        print(f"Result: {out}")

        # Condition the model based on the outputs
        cond_mdl = pyro.condition(curr_mdl, {'Scale-State': tc.tensor(out)})

        sampler = pyro.infer.Importance(cond_mdl, num_samples=10000)
        results = sampler.run(candidate_designs[best])
        normed_w = sampler.get_normalized_weights()
        resamples = tc.distributions.Categorical(normed_w).sample((10000,))
        sampled_vals = {}
        sampled_vals['BH'] = [results.exec_traces[s].nodes['BH']['value'] for s in resamples]

        # Update the current model based on the posterior
        sampled_vals = np.array([results.exec_traces[s].nodes[f"BH"]['value'] for s in resamples])
        for i in range(8):
            ball_priors[i] = np.count_nonzero(sampled_vals == i) / sampled_vals.size
        print(f"Round {test} posterior probabilities for index of heavier ball: {ball_priors[:]}")


if __name__ == '__main__':
    riddle_solve()

