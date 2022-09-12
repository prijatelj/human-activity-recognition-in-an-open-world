"""Prototype / dev work on  understanding and implementing the DPGMM.
Following the tutorial from pyro docs:
https://pyro.ai/examples/dirichlet_process_mixture.html
"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
#from torch.distributions.constraint_registry import transform_to #, biject_to

import pyro
from pyro.distributions import *
from pyro.infer import (
    Predictive,
    SVI,
    Trace_ELBO,
    MCMC,
    NUTS,
)
from pyro.infer.mcmc.util import initialize_model, summary
from pyro.optim import Adam


pyro.set_rng_seed(0)


# To run this on cuda... need to make sure all tensors are on gpu.
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.multiprocessing.set_sharing_strategy("file_system")


# Gens the toy data to work on
srcs = (
    MultivariateNormal(-8 * torch.ones(2), torch.eye(2)),
    MultivariateNormal(8 * torch.ones(2), torch.eye(2)),
    MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)),
    MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)),
)
src_means = torch.stack([s.mean for s in srcs])
n_samples_per_src = 50
data = torch.cat([mvn.sample([n_samples_per_src]) for mvn in srcs])

plt.scatter(data[:, 0], data[:, 1])
plt.title("Data Samples from Mixture of 4 Gaussians")
plt.show()
# Number of samples
N = data.shape[0]

# Number of components
T = 6


def mix_weights(beta):
    """Defines the "stick-breaking" function to generate the weights.

    Q: What _is_ beta?

    Args
    ----
    beta :
        samples of beta
    """
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


def model(data):
    """The "model".

    pyro.plates are setting up all computation to be assumed as conditionally
    independent.
    """
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample(
            "mu",
            MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)),
        )

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)


def guide(data):
    """The guide used by the Stochastic Variational Inference.

    Q: What _is_ a guide?
    """
    kappa = pyro.param(
        'kappa',
        lambda: Uniform(0, 2).sample([T-1]),
        constraint=constraints.positive,
    )
    tau = pyro.param(
        'tau',
        lambda: MultivariateNormal(torch.zeros(2), 3*torch.eye(2)).sample([T])
    )
    phi = pyro.param(
        'phi',
        lambda: Dirichlet(1/T * torch.ones(T)).sample([N]),
        constraint=constraints.simplex,
    )

    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

    with pyro.plate("mu_plate", T):
        q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))


optim = Adam({"lr": 0.05})

losses = []

# Stochastic Variational Inference (SVI)
svi = SVI(model, guide, optim, loss=Trace_ELBO())

def train(num_iterations):
    pyro.clear_param_store()
    for j in tqdm(range(num_iterations)):
        loss = svi.step(data)
        losses.append(loss)


def truncate(alpha, centers, weights):
    threshold = alpha**-1 / 100.
    thresh_mask = weights > threshold
    true_centers = centers[thresh_mask]
    true_weights = weights[thresh_mask] / torch.sum(weights[thresh_mask])
    return true_centers, true_weights

alpha = 0.1
train(1000)

# We make a point-estimate of our model parameters using the posterior means of
# tau and phi for the centers and weights
Bayes_Centers_01, Bayes_Weights_01 = truncate(
    alpha,
    pyro.param("tau").detach(),
    torch.mean(pyro.param("phi").detach(), dim=0),
)

alpha = 1.5
train(1000)

# We make a point-estimate of our model parameters using the posterior means of
# tau and phi for the centers and weights
Bayes_Centers_15, Bayes_Weights_15 = truncate(
    alpha,
    pyro.param("tau").detach(),
    torch.mean(pyro.param("phi").detach(), dim=0)
)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color="blue")
plt.scatter(Bayes_Centers_01[:, 0], Bayes_Centers_01[:, 1], color="red")
plt.scatter(src_means[:, 0], src_means[:, 1], color="green", marker='x')

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], color="blue")
plt.scatter(Bayes_Centers_15[:, 0], Bayes_Centers_15[:, 1], color="red")
plt.scatter(src_means[:, 0], src_means[:, 1], color="green", marker='x')
plt.tight_layout()
plt.show()


# Trying NUTS instead of SVI (takes longer, but the centers seem nicer.)
#n_chains = 5
#init_params, potential_fn, transforms, prototype_trace = initialize_model(
#    model,
#    model_args=(data),
#    num_chains=n_chains,
#)

nuts_kernel = NUTS(model, adapt_step_size=True)

mcmc = MCMC(
    nuts_kernel,
    num_samples=int(5e2),
    warmup_steps=int(1e3),
    #num_chains=n_chains, # NOTE does not seem to work on my lab machine.
    #initial_params=init_params,
    #transforms=transforms,
)

# 'Train' / fit the MCMC to data.
pyro.clear_param_store()
mcmc.run(data)
samples = mcmc.get_samples()

Bayes_Centers_NUTS = truncate(
    alpha,
    samples['mu'].mean(0),
    samples['beta'].mean(0),
)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color="blue")
plt.scatter(Bayes_Centers_01[:, 0], Bayes_Centers_01[:, 1], color="red", marker='o')
plt.scatter(src_means[:, 0], src_means[:, 1], color="green", marker='x')

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], color="blue")
plt.scatter(Bayes_Centers_NUTS[:, 0], Bayes_Centers_NUTS[:, 1], color="red", marker='o')
plt.scatter(src_means[:, 0], src_means[:, 1], color="green", marker='x')
plt.tight_layout()
plt.show()

# TODO Set a min number on beta s.t. num betas in [known_classes, n_observations]
# TODO Prefit MVNs per class (easy) and then use them in comp of DPGMM.


def dpgmm_model(
    data,
    alpha=1.0,
    max_comps=None,
    scale_type=None,
    known_comps=None,
):
    """Dirichelt Process Gaussian Mixture Model that estimates the full
    covariance matrix per component's MultivariateNormal. This follows the
    stick-breaking method [1] for fitting Dirichlet Process.

    Args
    ----
    data : torch.Tensor
        The data as observations of events in feature space
    alpha : float
        The hyperparameter for controlling indirectly how many components will
        be usd in final result.
    known_comps : list(MultivariateNormal)
        The list of known components as MultivariateNormal distributions.
    max_comps : int = None
        The total number of allowed maximum components to be estimated by the
        Dirichlet Process. Defaults to the number of data samples (rows).
    scale : str = None
        If given, indicates how the scale of the components' multivariate
        normal covariance matrix.

        'covariance_matrix' attempts to fit the entire covariance matrix.
        'diag' attempts to fit only the diagonal values of the matrix.
        'scaled_identity' attempts to fit a scalar parameter to the identity.


    Notes
    -----
    Citations:
    [1]
    @article{sethuraman_constructive_1994,
        title = {A {CONSTRUCTIVE} {DEFINITION} {OF} {DIRICHLET} {PRIORS}},
        volume = {4},
        issn = {1017-0405},
        url = {https://www.jstor.org/stable/24305538},
        number = {2},
        urldate = {2022-08-25},
        journal = {Statistica Sinica},
        author = {Sethuraman, Jayaram},
        year = {1994},
        note = {Publisher: Institute of Statistical Science, Academia Sinica},
        pages = {639--650},
    }
    """
    frepr_dim = data.shape[-1]
    dtype_info = torch.finfo(data.dtype)

    # Both the number of observations  and by default the max components
    n_obs = data.shape[0]
    if max_comps is None:
        max_comps = n_obs

    with pyro.plate('beta_plate', max_comps-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    # Specify component priors as Maximum Entropy Principle priors (Uniform)
    with pyro.plate('loc_plate', max_comps * frepr_dim):
        loc = pyro.sample(
            'loc',
            Uniform(dtype_info.resolution * -9, dtype_info.resolution * 9),
        ).reshape(max_comps, frepr_dim)

    if scale_type in {None, 'covariance_matrix'}:
        # Use LKJCholesky and batch matmul via einsum to sample positive definite
        # matrices for the covariance matrix (scale).
        with pyro.plate('scale_plate', max_comps):
            cov_factor = pyro.sample(
                'cov_factor',
                LKJCholesky(frepr_dim, concentration=1.0),
            )
            scale = torch.einsum('bij,bkj->bik', cov_factor, cov_factor)
        # NOTE there is scale_tril, which seems to be for the Cholesky factor
    elif scale_type == 'diag_scalar':
        with pyro.plate('scale_plate', max_comps):
            diag_scalar = pyro.sample(
                'diag_scalar',
                #Uniform(0, dtype_info.resolution * 9),
                HalfCauchy(scale=25 * torch.ones(1)),
            )
            scale = torch.einsum(
                'bi,bkj->ikj',
                diag_scalar.reshape(1,-1),
                torch.eye(frepr_dim).unsqueeze(0),
            )
    else:
        scale = torch.eye(frepr_dim)

    with pyro.plate('data', n_obs):
        z = pyro.sample('z', Categorical(mix_weights(beta)))
        if scale_type == 'eye':
            return pyro.sample(
                'obs',
                MultivariateNormal(loc[z], scale),
                obs=data,
            )
        return pyro.sample(
            'obs',
            MultivariateNormal(loc[z], scale[z]),
            obs=data,
        )


def dpgmm_posterior(beta, cov_factor, loc, max_comps, frepr_dim, alpha=1.0):
    """The expected posterior parameters of the Dirichlet Process Gaussian
    Mixture Model given the samples from the posterior distribution resulting
    from fitting an MCMC to dpggm_model().
    """
    expected_betas = mix_weights(beta.mean(0))

    # Truncate based on alpha
    threshold = alpha**-1 / 100.
    thresh_mask = expected_betas > threshold

    expected_betas = expected_betas[thresh_mask]
    expected_locs = loc.reshape(loc.shape[0], max_comps, frepr_dim).mean(0)[
        thresh_mask
    ]
    expected_cov_factor = cov_factor.mean(0)[thresh_mask]

    return (
        expected_betas,
        expected_locs,
        torch.einsum('bij,bkj->bik', expected_cov_factor, expected_cov_factor),
    )


def more_params_guide(categories_init=None):
    """Potential guide to be used for SVI version of more_params_model. Note
    that reading and digesting what exactly is SVI and the guide is important
    to finish this.
    """
    raise NotImplementedError()
    # TODO specify params
    beta = pyro.param(
        'beta',
        lambda: Uniform(0.0, 1.0).sample([max_comps]),
        constraint=constraints.positive,
    )
    loc = pyro.param(
        'loc',
        lambda: Uniform(
            dtype_info.resolution * -9,
            dtype_info.resolution * 9,
        ).sample([frepr_dim]),
    )
    scale = pyro.param(
        'scale',
        lambda: Uniform(
            dtype_info.resolution * -9,
            dtype_info.resolution * 9,
        ).sample([frepr_dim, frepr_dim]),
        constraint=constraints.positive_semidefinite,
    )
    categories = pyro.param(
        'categories',
        lambda: Dirichlet(
            1 / max_comps * torch.ones(max_comps)
        ).sample([n_obs]) if categories_init is None else categories_init,
        constraint=constraints.simplex,
    )
