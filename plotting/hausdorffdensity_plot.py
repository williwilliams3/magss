import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.random as jr
import jax.numpy as jnp
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


def make_reparametrization_plot(metric_type, make_ylabel=True):
    model_rng_key = jr.key(0)
    model_type = "twogaussians"
    sub_name = ""
    dim = 1
    manifold_config = ManifoldConfig(model_rng_key, model_type, sub_name)
    solver = set_solver(manifold_config)

    model, dim = set_model(manifold_config, dim)

    initial_x = jnp.zeros(dim)

    if metric_type in ["monge", "inverse_monge"]:
        param_values = [0.0, 0.001, 0.01, 0.1]
        param_greek = r"$\alpha^2$"

    elif metric_type in ["generative", "inverse_generative"]:

        param_values = [100.0, 1.0, 0.1, 0.01]
        param_greek = r"$\lambda$"

    y_values_repar = []
    for param in param_values:

        args = get_args_from_metric_params(metric_type, model, param)
        metric = set_metric(metric_type, dim, step_size_ode, solver, args)
        velocity = metric.sample_unit_ball(model_rng_key, initial_x)
        velocity = jnp.abs(velocity)

        def haussdorff_density(x):
            return model.logdensity_fn(x) - 0.5 * metric.log_determinant_fn(x)

        def logdensity_hausdorff_fn(theta):
            x_new, _ = metric.geodesic_fn(initial_x, velocity, theta)
            logdensity = haussdorff_density(x_new)
            return logdensity

        xlim = [-2.1, 2.1]
        x = jnp.linspace(xlim[0], xlim[1], 200)
        y2 = jax.vmap(logdensity_hausdorff_fn)(x)
        y2 = jnp.exp(y2)
        y_values_repar.append(y2)

    # Create the plots
    fig, axs = plt.subplots(1, 1, figsize=(4, 2), sharey=True)
    for i, param in enumerate(param_values):
        axs.plot(
            x,
            y_values_repar[i],
            label=f"{param_greek}={param}",
            color=palette[i],
            linewidth=2,
        )
    axs.set_xlim(xlim)
    axs.legend(fontsize=12)
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks
    plt.xlabel(r"$t$", fontsize=16)
    if make_ylabel:
        plt.ylabel(
            r"$p_{\mathcal{H}} (\hat{\gamma}_{(\boldsymbol{x}, \boldsymbol{v})}(t))$",
            fontsize=16,
        )

    # Add padding and show the plot
    plt.tight_layout()
    plt.savefig(f"figs/{metric_type}_reparametrization.png", dpi=200)
    plt.close()
    print(f"Saved to figs/{metric_type}_reparametrization.png")


if __name__ == __name__:
    make_reparametrization_plot("inverse_monge")
    make_reparametrization_plot("inverse_generative", make_ylabel=False)
