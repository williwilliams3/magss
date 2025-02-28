import jax
import jax.numpy as jnp
from .metrics import Metric


class InverseGenerative(Metric):

    def __init__(self, dim, step_size_ode, solver, kwargs):
        super().__init__(
            dim,
            step_size_ode,
            solver,
            set_metric_functions=set_inverse_generative,
            kwargs=kwargs,
        )


def set_inverse_generative(logdensity_fn, p_0, lambd, dim):

    def logmetric_fn(x):
        return 2 * (jnp.log((lambd + jnp.exp(logdensity_fn(x)))) - jnp.log(p_0 + lambd))

    def christoffel_fn(x, v):
        grad_logf = jax.grad(logmetric_fn)(x)
        dot_product = jnp.dot(v, grad_logf)
        norm_v_squared = jnp.square(v).sum()
        result = dot_product * v - 0.5 * norm_v_squared * grad_logf
        return result

    def squared_norm_fn(x, v):
        logfactor = logmetric_fn(x)
        return jnp.exp(logfactor) * jnp.square(v).sum()

    def inverse_sqrt_metric_fn(x):
        factor = jnp.exp(-0.5 * logmetric_fn(x))
        return factor

    def log_determinant_fn(x):
        logfactor = logmetric_fn(x)
        return dim * logfactor

    return christoffel_fn, squared_norm_fn, inverse_sqrt_metric_fn, log_determinant_fn
