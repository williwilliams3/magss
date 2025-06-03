import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


from metrics.models import TwoGaussians, Rosenbrock
from metrics import (
    InverseMonge,
    InverseGenerative,
    Euclidean,
    Monge,
    Generative,
)
from plotting.plotting_functions import get_contours


dim = 2


def plot_geodesic_balls(
    dim=2,
    metric_type="monge",
    model_type="twogaussians",
    x=-0.8 * jnp.ones(dim),
    num_points=5,
    angle_range=jnp.pi / 8,
    color="red",
):
    """Plots geodesic balls and geodesic lines for a given metric."""

    lw = 2
    if model_type == "twogaussians":
        model = TwoGaussians(dim)
    elif model_type == "rosenbrock":
        model = Rosenbrock(dim)
    logdensity_fn = model.logdensity_fn

    # Metric setup
    step_size_ode = None
    solver = diffrax.Dopri5()

    xlim = [-1.5, 1.5]
    ylim = [-1.5, 1.5]

    # x = jnp.zeros(dim)

    if metric_type == "inverse_monge":
        kwargs = {"logdensity_fn": logdensity_fn, "alpha2": 0.001}
        metric = InverseMonge(dim, step_size_ode, solver, kwargs)
        inverse_sqrt_metric_fn = metric.inverse_sqrt_metric_fn

    elif metric_type == "inverse_generative":
        kwargs = {"logdensity_fn": logdensity_fn, "p_0": 1.0, "lambd": 1.0, "dim": dim}
        metric = InverseGenerative(dim, step_size_ode, solver, kwargs)
        inverse_sqrt_metric_fn = lambda x: metric.inverse_sqrt_metric_fn(x) * jnp.eye(
            dim
        )

    elif metric_type == "monge":
        kwargs = {"logdensity_fn": logdensity_fn, "alpha2": 0.1}
        metric = Monge(dim, step_size_ode, solver, kwargs)
        inverse_sqrt_metric_fn = metric.inverse_sqrt_metric_fn

    elif metric_type == "generative":
        kwargs = {"logdensity_fn": logdensity_fn, "p_0": 1.0, "lambd": 1.0, "dim": dim}
        metric = Generative(dim, step_size_ode, solver, kwargs)
        inverse_sqrt_metric_fn = lambda x: metric.inverse_sqrt_metric_fn(x) * jnp.eye(
            dim
        )

    elif metric_type == "euclidean":
        metric = Euclidean(dim)
        inverse_sqrt_metric_fn = lambda x: jnp.eye(dim)

    # Get true distribution contours
    true_dist_levels = model.true_dist_levels
    contours = get_contours(model.xlim, model.ylim, logdensity_fn)
    [X, Y, Z] = contours

    # Set up the figure
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)

    num_points = num_points + num_points % 2  # make sure it's even
    mid_angle = jnp.pi / 4
    angles = jnp.linspace(
        mid_angle - angle_range, mid_angle + angle_range, num_points, endpoint=False
    )
    directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
    # velocities = unit_vectors @ inverse_sqrt_metric_fn(x)
    # Plot unit vector in red
    for direction in directions:
        direction /= jnp.linalg.norm(direction)  # Normalize to unit length
        # Plot geodesic line in forward and backward directions
        velocity = inverse_sqrt_metric_fn(x) @ direction
        times = jnp.linspace(-2, 6, 300)
        geodesic_fn_ = jax.jit(metric.geodesic_fn)
        geodesic_samples, geodesic_velocities = jax.vmap(
            geodesic_fn_, in_axes=(None, None, 0)
        )(x, velocity, times)

        speeds = jnp.log(jnp.linalg.norm(geodesic_velocities, axis=1) ** 2)
        # print("speeds", speeds)

        # Create segments for coloring
        points = geodesic_samples[:, :2].reshape(-1, 1, 2)
        segments = jnp.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap="plasma",  # or 'viridis', 'inferno', etc.
            norm=plt.Normalize(speeds.min(), speeds.max()),
            array=speeds[:-1],  # one less because segments are between points
            linewidth=lw,
            alpha=0.9,
        )
        ax.add_collection(lc)

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
    # Plot starting point
    ax.scatter(x[0], x[1], c="black", s=100, marker="*", label="Center", zorder=15)

    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)  # Remove extra padding around the plot
    plt.tight_layout()  # Adjust layout to remove unnecessary white space
    plt.savefig(
        f"figs/geodesics/{metric_type}_{model_type}.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    print(f"Saved figure to figs/geodesics/{metric_type}_{model_type}.png")


if __name__ == "__main__":

    # Call the function to generate the plot
    plot_geodesic_balls(metric_type="euclidean", angle_range=jnp.pi / 8, color="blue")
    plot_geodesic_balls(
        metric_type="inverse_generative",
        angle_range=jnp.pi / 8,
        num_points=7,
        color="green",
    )
    plot_geodesic_balls(
        metric_type="inverse_monge", num_points=7, angle_range=jnp.pi / 3, color="red"
    )
