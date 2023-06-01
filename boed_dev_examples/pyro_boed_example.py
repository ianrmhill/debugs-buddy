import torch

import pyro
import pyro.distributions as dist


sensitivity = 1.0
prior_mean = torch.tensor(7.0)
prior_sd = torch.tensor(2.0)


def model(l):
    # Dimension -1 of `l` represents the number of rounds
    # Other dimensions are batch dimensions: we indicate this with a plate_stack
    with pyro.plate_stack("plate", l.shape[:-1]):
        theta = pyro.sample("theta", dist.Normal(prior_mean, prior_sd))
        # Share theta across the number of rounds of the experiment
        # This represents repeatedly testing the same participant
        theta = theta.unsqueeze(-1)
        # This defines a *logistic regression* model for y
        logit_p = sensitivity * (theta - l)
        # The event shape represents responses from the same participant
        y = pyro.sample("y", dist.Bernoulli(logits=logit_p).to_event(1))
        return y

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

# We sample five times from the prior
theta = (prior_mean + prior_sd * torch.randn((5,1)))
l = torch.arange(1, 16, dtype=torch.float)
# This is the same as using 'logits=' in the prior above
prob = torch.sigmoid(sensitivity * (theta - l))

plt.figure(figsize=(12, 8))
for curve in torch.unbind(prob, 0):
    plt.plot(l.numpy(), curve.numpy(), marker='o')
plt.xlabel("Length of sequence $l$")
plt.ylabel("Probability of correctly remembering\na sequence of length $l$")
plt.legend(["Person {}".format(i+1) for i in range(5)])
plt.show()


from torch.distributions.constraints import positive

def guide(l):
    # The guide is initialised at the prior
    posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
    posterior_sd = pyro.param("posterior_sd", prior_sd.clone(), constraint=positive)
    pyro.sample("theta", dist.Normal(posterior_mean, posterior_sd))

l_data = torch.tensor([5., 7., 9.])
y_data = torch.tensor([1., 1., 0.])


from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

conditioned_model = pyro.condition(model, {"y": y_data})
svi = SVI(conditioned_model,
          guide,
          Adam({"lr": .001}),
          loss=Trace_ELBO(),
          num_samples=100)
pyro.clear_param_store()
num_iters = 5000
for i in range(num_iters):
    elbo = svi.step(l_data)
    if i % 500 == 0:
        print("Neg ELBO:", elbo)

print("Prior:     N({:.3f}, {:.3f})".format(prior_mean, prior_sd))
print("Posterior: N({:.3f}, {:.3f})".format(pyro.param("posterior_mean"),
                                                pyro.param("posterior_sd")))


def marginal_guide(design, observation_labels, target_labels):
    # This shape allows us to learn a different parameter for each candidate design l
    q_logit = pyro.param("q_logit", torch.zeros(design.shape[-2:]))
    pyro.sample("y", dist.Bernoulli(logits=q_logit).to_event(1))


from pyro.contrib.oed.eig import marginal_eig

# The shape of `candidate_designs` is (number designs, 1)
# This represents a batch of candidate designs, each design is for one round of experiment
candidate_designs = torch.arange(1, 15, dtype=torch.float).unsqueeze(-1)
pyro.clear_param_store()
num_steps, start_lr, end_lr = 1000, 0.1, 0.001
optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                      'optim_args': {'lr': start_lr},
                                      'gamma': (end_lr / start_lr) ** (1 / num_steps)})

eig = marginal_eig(model,
                   candidate_designs,  # design, or in this case, tensor of possible designs
                   "y",  # site label of observations, could be a list
                   "theta",  # site label of 'targets' (latent variables), could also be list
                   num_samples=100,  # number of samples to draw per step in the expectation
                   num_steps=num_steps,  # number of gradient steps
                   guide=marginal_guide,  # guide q(y)
                   optim=optimizer,  # optimizer with learning rate decay
                   final_num_samples=10000  # at the last step, we draw more samples
                   # for a more accurate EIG estimate
                   )


plt.figure(figsize=(10,5))
matplotlib.rcParams.update({'font.size': 22})
plt.plot(candidate_designs.numpy(), eig.detach().numpy(), marker='o', linewidth=2)
plt.xlabel("$l$")
plt.ylabel("EIG($l$)")
plt.show()

best_l = 1 + torch.argmax(eig)
print("Optimal design:", best_l.item())

q_prob = torch.sigmoid(pyro.param("q_logit"))
print("   l | q(y = 1 | l)")
for (l, q) in zip(candidate_designs, q_prob):
    print("{:>4} | {}".format(int(l.item()), q.item()))



def synthetic_person(l):
    # The synthetic person can remember any sequence shorter than 6
    # They cannot remember any sequence of length 6 or above
    # (There is no randomness in their responses)
    y = (l < 6.).float()
    return y


def make_model(mean, sd):
    def model(l):
        # Dimension -1 of `l` represents the number of rounds
        # Other dimensions are batch dimensions: we indicate this with a plate_stack
        with pyro.plate_stack("plate", l.shape[:-1]):
            theta = pyro.sample("theta", dist.Normal(mean, sd))
            # Share theta across the number of rounds of the experiment
            # This represents repeatedly testing the same participant
            theta = theta.unsqueeze(-1)
            # This define a *logistic regression* model for y
            logit_p = sensitivity * (theta - l)
            # The event shape represents responses from the same participant
            y = pyro.sample("y", dist.Bernoulli(logits=logit_p).to_event(1))
            return y
    return model

ys = torch.tensor([])
ls = torch.tensor([])
history = [(prior_mean, prior_sd)]
pyro.clear_param_store()
current_model = make_model(prior_mean, prior_sd)

for experiment in range(10):
    print("Round", experiment + 1)

    # Step 1: compute the optimal length
    optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                          'optim_args': {'lr': start_lr},
                                          'gamma': (end_lr / start_lr) ** (1 / num_steps)})
    eig = marginal_eig(current_model, candidate_designs, "y", "theta", num_samples=100,
                       num_steps=num_steps, guide=marginal_guide, optim=optimizer,
                       final_num_samples=10000)
    best_l = 1 + torch.argmax(eig).float().detach()

    # Step 2: run the experiment, here using the synthetic person
    print("Asking the participant to remember a sequence of length", int(best_l.item()))
    y = synthetic_person(best_l)
    if y:
        print("Participant remembered correctly")
    else:
        print("Participant could not remember the sequence")
    # Store the sequence length and outcome
    ls = torch.cat([ls, best_l.expand(1)], dim=0)
    ys = torch.cat([ys, y.expand(1)])

    # Step 3: learn the posterior using all data seen so far
    conditioned_model = pyro.condition(model, {"y": ys})
    svi = SVI(conditioned_model,
              guide,
              Adam({"lr": .005}),
              loss=Trace_ELBO(),
              num_samples=100)
    num_iters = 2000
    for i in range(num_iters):
        elbo = svi.step(ls)

    history.append((pyro.param("posterior_mean").detach().clone().numpy(),
                    pyro.param("posterior_sd").detach().clone().numpy()))
    current_model = make_model(pyro.param("posterior_mean").detach().clone(),
                               pyro.param("posterior_sd").detach().clone())
    print("Estimate of \u03b8: {:.3f} \u00b1 {:.3f}\n".format(*history[-1]))

import numpy as np
from scipy.stats import norm
import matplotlib.colors as colors
import matplotlib.cm as cmx

matplotlib.rcParams.update({'font.size': 22})
cmap = plt.get_cmap('winter')
cNorm = colors.Normalize(vmin=0, vmax=len(history) - 1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
plt.figure(figsize=(12, 6))
x = np.linspace(0, 14, 100)
for idx, (mean, sd) in enumerate(history):
    color = scalarMap.to_rgba(idx)
    y = norm.pdf(x, mean, sd)
    plt.plot(x, y, color=color)
    plt.xlabel("$\\theta$")
    plt.ylabel("p.d.f.")
plt.show()

pyro.clear_param_store()
ls = torch.arange(1, 11, dtype=torch.float)
ys = synthetic_person(ls)
conditioned_model = pyro.condition(model, {"y": ys})
svi = SVI(conditioned_model,
          guide,
          Adam({"lr": .005}),
          loss=Trace_ELBO(),
          num_samples=100)
num_iters = 2000
for i in range(num_iters):
    elbo = svi.step(ls)

plt.figure(figsize=(12,6))
matplotlib.rcParams.update({'font.size': 22})
y1 = norm.pdf(x, pyro.param("posterior_mean").detach().numpy(),
              pyro.param("posterior_sd").detach().numpy())
y2 = norm.pdf(x, history[-1][0], history[-1][1])
plt.plot(x, y1)
plt.plot(x, y2)
plt.legend(["Simple design", "Optimal design"])
plt.xlabel("$\\theta$")
plt.ylabel("p.d.f.")
plt.show()

