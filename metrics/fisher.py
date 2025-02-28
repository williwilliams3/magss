import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular
from .metrics import Metric
from .geodesics import general_christoffel_fn
from .models import Distribution


class Fisher(Metric):

    def __init__(self, dim, step_size_ode, solver, kwargs):
        super().__init__(
            dim, step_size_ode, solver, set_metric_functions=set_fisher, kwargs=kwargs
        )


def set_fisher(distribution: Distribution):

    _metric_fn = distribution.fisher_metric_fn

    def squared_norm_fn(x, v):
        metric_vector = jnp.dot(_metric_fn(x), v)
        return jnp.dot(v, metric_vector)

    def inverse_sqrt_metric_fn(x):
        metric = _metric_fn(x)
        shape = jnp.shape(metric)[:1]
        L = cholesky(metric, lower=True)
        identity = jnp.identity(shape[0])
        metric_invsqrt = solve_triangular(L, identity, lower=True, trans=True)
        return metric_invsqrt

    def log_determinant_fn(x):
        return jnp.linalg.slogdet(_metric_fn(x))[1]

    def christoffel_fn(x, v):
        return general_christoffel_fn(_metric_fn, x, v)

    return christoffel_fn, squared_norm_fn, inverse_sqrt_metric_fn, log_determinant_fn
