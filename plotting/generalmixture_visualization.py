import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from metrics.models.general_mixture import (
    GeneralMixture,
)
from plotting import get_plot, get_contours

"""
Scatter plot of samples and linear separator
"""


num_samples = 1000
dim = 2
scale = 0.1 * jnp.ones(2)
loc = jnp.array([2.0, 2.0])

model = GeneralMixture(dim)
rng_key = jr.key(0)
# samples = model.sample(rng_key, 1000)
key = 1
indices = jr.choice(rng_key, 10_000, shape=(1000,), replace=False)
dir_samples = f"logs/model=generalmixture/sampler=meta_agss/dim=2/manifold.integrator=dopri8,manifold.metric=inverse_monge,manifold.metric_param=0.0001,manifold.step_size_ode=null,sampler.step_size=0.001,sampler.sweeps=5/rng_key={key}/samples.npy"
tensor_samples = np.load(dir_samples)
samples = tensor_samples.reshape(10_000, 2)
samples = samples[indices]
logdensity_fn = model.logdensity_fn

# Linear separator
x_points = jnp.array([-2, -1.8])
y_points = jnp.array([0, 1])

# Compute slope (m) and intercept (b) of the line: y = mx + b


# Generate x values for the line
x_points = jnp.array([-2, -1.8])
y_points = jnp.array([0, 1])
m = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
b = y_points[0] - m * x_points[0]
x_line = jnp.linspace(-2.5, -1.5, 100)  # Extending a bit for better visualization
y_line = m * x_line + b  # Line equation


figsize = (6, 3.5)
xlim = model.xlim
ylim = model.ylim
true_dist_levels = model.true_dist_levels
model_type = "generalfunnel"
contours = get_contours(xlim, ylim, logdensity_fn)
true_dist_colors = ["black", "black", "black"]
[X, Y, Z] = contours
plt.figure(figsize=figsize)
plt.plot(x_line, y_line, "r--", lw=3)  # Red line
plt.contour(
    X,
    Y,
    Z,
    levels=true_dist_levels,
    colors=true_dist_colors,
    linestyles="dashed",
    linewidths=1.5,
    alpha=0.5,
)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.xticks([])
plt.yticks([])
# plt.tight_layout()

plt.savefig("figs/generalmixture.png")
