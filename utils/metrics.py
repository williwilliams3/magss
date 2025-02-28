from metrics import (
    Euclidean,
    Monge,
    Generative,
    InverseMonge,
    InverseGenerative,
    Fisher,
)
import diffrax


def set_solver(manifold_config):
    solver = manifold_config.integrator
    if solver == "euler":
        return diffrax.Euler()
    elif solver == "tsit":
        return diffrax.Tsit5()
    elif solver == "dopri5":
        return diffrax.Dopri5()
    elif solver == "dopri8":
        return diffrax.Dopri8()
    elif solver == "kv3":
        return diffrax.Kvaerno3()
    elif solver == "kv5":
        return diffrax.Kvaerno5()
    elif solver == "revheun":
        return diffrax.ReversibleHeun()
    else:
        return NotImplementedError(f"Invalid solver {solver}")


def get_args_from_metric_params(metric_type, distribution, metric_param):
    args = {}
    logdensity_fn = distribution.logdensity_fn
    # Set the metric parameters
    if metric_type in ["monge", "inverse_monge"]:
        args["alpha2"] = metric_param
        args["logdensity_fn"] = logdensity_fn
    elif metric_type in ["generative", "inverse_generative"]:
        args["lambd"] = metric_param
        args["p_0"] = 1.0
        args["logdensity_fn"] = logdensity_fn
        args["dim"] = distribution.dim
    elif metric_type == "fisher":
        args["distribution"] = distribution
    return args


def set_metric(metric_type, dim, step_size_ode, solver, kwargs):
    if metric_type == "euclidean":
        metric = Euclidean(dim)
    elif metric_type == "monge":
        metric = Monge(dim, step_size_ode, solver, kwargs)
    elif metric_type == "generative":
        metric = Generative(dim, step_size_ode, solver, kwargs)
    elif metric_type == "inverse_monge":
        metric = InverseMonge(dim, step_size_ode, solver, kwargs)
    elif metric_type == "inverse_generative":
        metric = InverseGenerative(dim, step_size_ode, solver, kwargs)
    elif metric_type == "fisher":
        metric = Fisher(dim, step_size_ode, solver, kwargs)
    else:
        return NotImplementedError

    return metric
