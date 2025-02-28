import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_model, set_metric, set_solver, get_args_from_metric_params


palette = sns.color_palette("Set2", 4)


@dataclass
class ManifoldConfig:
    model_rng_key: int
    model: str
    sub_name: str
    integrator: str = "dopri5"


step_size_ode = None
solver = "dopri5"


def set_logdensity_fn(metric_type, param):
    model_rng_key = jr.key(0)
    model_type = "twogaussians"
    sub_name = ""
    dim = 1
    manifold_config = ManifoldConfig(model_rng_key, model_type, sub_name)
    solver = set_solver(manifold_config)

    model, dim = set_model(manifold_config, dim)
    model.means = [jnp.ones(dim) * -1.5, jnp.ones(dim) * 1.5]

    args = get_args_from_metric_params(metric_type, model, param)
    metric = set_metric(metric_type, dim, step_size_ode, solver, args)

    x_ini = jnp.array([-1.53])
    velocity = metric.sample_unit_ball(model_rng_key, x_ini)
    velocity = jnp.abs(velocity)

    def haussdorff_density(x):
        return model.logdensity_fn(x) - 0.5 * metric.log_determinant_fn(x)

    @jax.jit
    def logdensity_hausdorff_fn(theta):
        x_new, _ = metric.geodesic_fn(x_ini, velocity, theta)
        logdensity = haussdorff_density(x_new)
        return logdensity

    return logdensity_hausdorff_fn, x_ini


def set_color(s, y):
    if y > s:
        return "green"
    else:
        return "red"


def make_slice_plot(
    logdensity_fn,
    x_ini,
    color,
    label,
    param,
    right_steps=1,
    left_steps=1,
):

    xlim = jnp.array([-2.5, 1.7]) - x_ini
    x = jnp.linspace(xlim[0], xlim[1], 200)
    y2 = jax.vmap(logdensity_fn)(x)
    y_values = jnp.exp(y2)

    # Create the plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    ax.plot(x, y_values, color=color, linewidth=2, label=label)

    # Step out procedure
    t_ini = 0.0
    w = 0.4
    u = 0.3

    l = t_ini - u * w
    r = l + w
    y_ini = jnp.exp(logdensity_fn(t_ini))
    s = 0.6 * y_ini
    ax.vlines(t_ini, 0, y_ini, color="gray", linewidth=2)
    ax.scatter([t_ini], [s], color="red", s=100, marker="+", zorder=10)

    pl = jnp.exp(logdensity_fn(l))
    pr = jnp.exp(logdensity_fn(r))
    ax.scatter(l, 0.0, color=set_color(s, pl), marker="X", zorder=10)
    ax.scatter(r, 0.0, color=set_color(s, pr), marker="X", zorder=10)
    ax.scatter([l], [pl], color=set_color(s, pl), s=100, marker=".", zorder=10)
    ax.scatter([r], [pr], color=set_color(s, pr), s=100, marker=".", zorder=10)

    for _ in range(left_steps - 1):
        l = l - w
        pl = jnp.exp(logdensity_fn(l))
        ax.scatter(l, 0.0, color=set_color(s, pl), marker="X", zorder=10)
        ax.scatter([l], [pl], color=set_color(s, pl), s=100, marker=".", zorder=10)

    for _ in range(right_steps - 1):
        r = r + w
        pr = jnp.exp(logdensity_fn(r))
        ax.scatter(r, 0.0, color=set_color(s, pr), marker="X", zorder=10)
        ax.scatter([r], [pr], color=set_color(s, pr), s=100, marker=".", zorder=10)

    ax.hlines(0, l, r, color="red", linewidth=2)

    # Add annotations
    ax.text(t_ini + 0.05, s, "s", fontsize=14, color="black")
    ax.text(t_ini, -0.2 * y_ini, r"$t=0$", fontsize=10, color="black", ha="center")
    ax.text(r, -0.2 * y_ini, r"$r$", fontsize=14, color="black", ha="center")
    ax.text(l, -0.2 * y_ini, r"$\ell$", fontsize=14, color="black", ha="center")
    ax.text(r + 0.05, pr, r"$p(r)$", fontsize=14, color="black")  # Right red dot
    ax.text(l - 0.3, pl, r"$p(\ell)$", fontsize=14, color="black")  # Left red dot

    # Fill the region where p(t) > p(s)
    # ax.fill_between(x, y_values, where=(y_values > s), color="lightgrey", alpha=0.5)
    ax.fill_between(x, y_values, s, where=(y_values > s), color="lightgrey", alpha=0.5)

    ax.set_xlim(xlim)
    legend_ax = ax.legend(fontsize=20, loc="upper right")
    legend_ax.get_frame().set_linewidth(0)
    ax.set_frame_on(False)
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks

    plt.tight_layout()
    # Create dir if not exists
    if not os.path.exists("figs/visualization"):
        os.makedirs("figs/visualization")

    plt.savefig(f"figs/visualization/{metric_type}_{param}.png", dpi=200)
    plt.close()
    print(f"Saved to figs/visualization/{metric_type}_{param}.png")


if __name__ == "__main__":
    # Make Figure 1 in plot
    params = ["", 1.0, 0.1, 0.01]
    right_steps = [1, 2, 2, 3]
    left_steps = [1, 2, 2, 2]
    for i in range(len(params)):
        param = params[i]
        if i == 0:
            metric_type = "euclidean"
            logdensity_fn, x_ini = set_logdensity_fn("euclidean", param)
            label = "Euclidean"
        else:
            metric_type = "inverse_generative"
            logdensity_fn, x_ini = set_logdensity_fn(metric_type, param)
            label = rf"$\lambda={param}$"
        make_slice_plot(
            logdensity_fn,
            x_ini,
            palette[i],
            label,
            param,
            right_steps=right_steps[i],
            left_steps=left_steps[i],
        )
