import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import matplotlib.pyplot as plt
from metrics.models import Funnel, TwoGaussians
from metrics import Monge, Generative, InverseMonge, InverseGenerative, Euclidean
from plotting.plotting_functions import get_contours
import seaborn as sns


def plot_geodesic_balls(dim=2, num_points=100, radii=[1], metric_type="monge"):
    """Plots geodesic balls and geodesic lines for a given metric."""
    lw = 4
    palette = sns.color_palette("Set2", len(radii))

    model = Funnel(dim)
    logdensity_fn = model.logdensity_fn

    # Metric setup
    step_size_ode = None
    solver = diffrax.Dopri5()

    if metric_type == "monge":
        xlim = [-5, 5]
        ylim = [-5, 2.66]
        x = jnp.ones(dim)
        kwargs = {"logdensity_fn": logdensity_fn, "alpha2": 1.0}
        metric = Monge(dim, step_size_ode, solver, kwargs)
        inverse_sqrt_metric_fn = metric.inverse_sqrt_metric_fn

    elif metric_type == "generative":
        xlim = [-5, 5]
        ylim = [-5, 2.66]
        x = jnp.ones(dim)
        kwargs = {"logdensity_fn": logdensity_fn, "p_0": 0.1, "lambd": 0.1, "dim": dim}
        metric = Generative(dim, step_size_ode, solver, kwargs)
        inverse_sqrt_metric_fn = lambda x: metric.inverse_sqrt_metric_fn(x) * jnp.eye(
            dim
        )

    elif metric_type == "euclidean":
        xlim = [-5, 5]
        ylim = [-5, 2.66]
        x = jnp.ones(dim)
        metric = Euclidean(dim)
        inverse_sqrt_metric_fn = lambda x: jnp.eye(dim)

    # Compute unit circle vectors
    angles = jnp.linspace(0, 2 * jnp.pi, num_points, endpoint=False)
    unit_vectors = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)

    # Get true distribution contours
    true_dist_levels = model.true_dist_levels
    contours = get_contours(model.xlim, model.ylim, logdensity_fn)
    [X, Y, Z] = contours

    # Set up the figure
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)

    for i, radius in enumerate(radii):
        # Compute geodesic ball for given radius
        velocities = unit_vectors @ inverse_sqrt_metric_fn(x)
        geodesic_fn_ = jax.jit(metric.geodesic_fn)
        samples, _ = jax.vmap(geodesic_fn_, in_axes=(None, 0, None))(
            x, velocities, radius
        )

        # Plot geodesic ball
        ax.plot(
            samples[:, 0], samples[:, 1], c=palette[i], label=f"Radius {radius}", lw=lw
        )

        # Connect last point to first point
        ax.plot(
            [samples[-1, 0], samples[0, 0]],
            [samples[-1, 1], samples[0, 1]],
            c=palette[i],
            lw=lw,
        )

    # Plot unit vector in red
    direction = jnp.array([-0.8, -1.0])
    direction /= jnp.linalg.norm(direction)  # Normalize to unit length
    ax.quiver(
        x[0],
        x[1],
        direction[0],
        direction[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.01,
        label="Unit Vector",
    )

    # Plot geodesic line in forward and backward directions
    velocity = inverse_sqrt_metric_fn(x) @ direction
    times = jnp.linspace(-2, 7, 50)
    geodesic_fn_ = jax.jit(metric.geodesic_fn)
    # geodesic_samples, _ = geodesic_fn_(x, inverse_sqrt_metric_fn(x) @ direction, times)
    geodesic_samples, _ = jax.vmap(geodesic_fn_, in_axes=(None, None, 0))(
        x, velocity, times
    )

    ax.plot(
        geodesic_samples[:, 0],
        geodesic_samples[:, 1],
        "r--",
        lw=lw,
        label="Geodesic Line",
    )

    ax.scatter(x[0], x[1], c="black", s=100, marker="*", label="Center")

    # Plot true density contours
    ax.contour(
        X,
        Y,
        Z,
        levels=true_dist_levels,
        colors="black",
        linestyles="dashed",
        linewidths=1.5,
        alpha=0.7,
    )

    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)  # Remove extra padding around the plot
    plt.tight_layout()  # Adjust layout to remove unnecessary white space
    # Make directory if not exits
    if not os.path.exists("figs"):
        os.makedirs("figs")
    plt.savefig(f"figs/geodesic_balls_{metric_type}_hitandrun.png")
    print(f"Saved figure to figs/geodesic_balls_{metric_type}_hitandrun.png")


# Call the function to generate the plot
plot_geodesic_balls(dim=2, radii=[1.0], metric_type="monge")
plot_geodesic_balls(radii=[1.0], metric_type="euclidean", num_points=100)
