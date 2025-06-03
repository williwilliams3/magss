import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(current_file_path))

import jax.numpy as jnp
import blackjax
import mcmc
from utils import set_metric, set_solver, get_args_from_metric_params


def get_sampler(
    distribution,
    manifold_config,
    sampler_config,
):
    # Extract the necessary information
    dim = manifold_config.dim
    metric_type = manifold_config.metric
    metric_param = manifold_config.metric_param
    step_size_ode = manifold_config.step_size_ode
    sampler_type = sampler_config.sampler_type
    sub_sampler_type = sampler_config.sub_sampler_type
    step_size = sampler_config.step_size
    logdensity_fn = distribution.logdensity_fn

    if sampler_type == "hmc" or sub_sampler_type == "hmc":
        num_integration_steps = sampler_config.num_integration_steps
        sampler_fn = lambda logdensity_fn: blackjax.hmc(
            logdensity_fn,
            step_size=step_size,
            num_integration_steps=num_integration_steps,
            inverse_mass_matrix=jnp.ones(dim),
        )
    elif sampler_type == "mala" or sub_sampler_type == "mala":
        sampler_fn = lambda logdensity_fn: blackjax.mala(
            logdensity_fn, step_size=step_size
        )
    elif sampler_type == "rmh":
        sampler_fn = lambda logdensity_fn: blackjax.normal_random_walk(
            logdensity_fn, sampler_config.ts[0]
        )
    elif sampler_type == "nuts":
        # No window Adaptation
        sampler_fn = lambda logdensity_fn: blackjax.nuts(
            logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=jnp.ones(dim),
        )
    elif sampler_type in ["rla", "magss", "map_magss"]:
        solver = set_solver(manifold_config)
        args = get_args_from_metric_params(metric_type, distribution, metric_param)
        metric = set_metric(
            metric_type,
            dim,
            step_size_ode,
            solver,
            kwargs=args,
        )

        # Set the sampler function
        if sampler_type in ["magss"]:

            def sampler_fn(logdensity_fn):
                return mcmc.geodesic_slice_sampler(
                    logdensity_fn,
                    step_size,
                    sampler_config.max_step_outs,
                    metric=metric,
                )

        elif sampler_type == "rla":

            def sampler_fn(logdensity_fn):
                return mcmc.riemannianlaplace(metric)

    else:
        raise ValueError("Invalid sampler name")

    if sampler_type == "digs":
        num_ts = sampler_config.num_time_steps
        alpha_min = sampler_config.alpha_min
        alpha_max = sampler_config.alpha_max
        ts = jnp.linspace(alpha_min, alpha_max, num_ts)
        gibbs_sweeps = sampler_config.sweeps
        alg_steps = sampler_config.alg_steps
        sampler = mcmc.digs(
            logdensity_fn=logdensity_fn,
            sampler_fn=sampler_fn,
            num_alphas=num_ts,
            alphas=ts,
            gibbs_sweeps=gibbs_sweeps,
            alg_steps=alg_steps,
            get_proposal=True,
        )

    elif sampler_type == "pt":
        num_temperatures = sampler_config.num_temperatures
        b_min = sampler_config.b_min
        inv_temperatures = b_min ** (
            jnp.arange(0, num_temperatures) / (num_temperatures - 1)
        )
        alg_steps = sampler_config.alg_steps
        sampler = mcmc.pt(
            logdensity_fn=logdensity_fn,
            num_temperatures=num_temperatures,
            inv_temperatures=inv_temperatures,
            sampler_fn=sampler_fn,
            alg_steps=alg_steps,
        )
    elif sampler_type == "meta_magss":
        solver = set_solver(manifold_config)
        args = get_args_from_metric_params(metric_type, distribution, metric_param)
        metric = set_metric(
            metric_type,
            dim,
            step_size_ode,
            solver,
            kwargs=args,
        )
        sampler = mcmc.meta_geodesic_slice_sampler(
            logdensity_fn=logdensity_fn,
            sampler_fn=sampler_fn,
            alg_steps=sampler_config.alg_steps,
            step_size=sampler_config.step_size_meta,
            max_stepouts=sampler_config.max_step_outs,
            metric=metric,
            sweeps=sampler_config.sweeps,
        )
    else:
        sampler = sampler_fn(logdensity_fn)
    return sampler
