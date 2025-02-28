import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular
from metrics.geodesics import numerical_integrate_geodesic_fn, general_christoffel_fn
from metrics import Monge, Metric


class MongeFullrank(Metric):

    def __init__(self, dim: int, logdensity_fn, alpha2):
        super().__init__(dim)
        (
            self.christoffel_fn,
            self.squared_norm_fn,
            self.inverse_sqrt_metric_fn,
            self.log_determinant_fn,
        ) = set_monge_fullrank(logdensity_fn, alpha2)

    def sample_unit_ball(self, rng_key, x):
        v = self.sample_velocity(rng_key, x)
        squared_norm = self.squared_norm_fn(x, v)
        v /= jnp.sqrt(squared_norm)
        return v

    def sample_velocity(self, rng_key, x):
        v = jr.normal(rng_key, x.shape)
        inverse_sqrt_metric = self.inverse_sqrt_metric_fn(x)
        return jnp.dot(inverse_sqrt_metric, v)

    def geodesic_fn(self, x, v, t):
        dim = self.dim
        output = numerical_integrate_geodesic_fn(dim, self.christoffel_fn, x, v, t)
        return output[0, :dim], output[0, dim:]


def set_monge_fullrank(logdensity_fn, alpha2):
    grad_fn = jax.grad(logdensity_fn)

    def _metric_fn(x):
        grad = grad_fn(x)
        return jnp.eye(x.shape[0]) + alpha2 * jnp.outer(grad, grad)

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


def test_inverse_monge_classes():
    dim = 2
    alpha2 = 1.0
    rng_key = jr.PRNGKey(42)
    x = jnp.ones((dim,)) + 0.1 * jr.normal(rng_key, (dim,))

    # Dummy log-density function
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    # Initialize both classes
    fullrank_model = MongeFullrank(dim, logdensity_fn, alpha2)
    monge_model = Monge(dim, logdensity_fn, alpha2)

    # Test squared_norm_fn
    v = jr.normal(rng_key, x.shape)
    squared_norm_fullrank = fullrank_model.squared_norm_fn(x, v)
    squared_norm_monge = monge_model.squared_norm_fn(x, v)
    assert jnp.allclose(
        squared_norm_fullrank, squared_norm_monge, atol=1e-5
    ), f"squared_norm_fn outputs do not match {squared_norm_fullrank} != {squared_norm_monge}"

    # Test log_determinant_fn
    log_det_fullrank = fullrank_model.log_determinant_fn(x)
    log_det_monge = monge_model.log_determinant_fn(x)
    assert jnp.allclose(
        log_det_fullrank, log_det_monge, atol=1e-5
    ), f"log_determinant_fn outputs do not match {log_det_fullrank} != {log_det_monge}"

    # Test cholensky decomposition
    L_fullrank = fullrank_model.inverse_sqrt_metric_fn(x)
    L = monge_model.inverse_sqrt_metric_fn(x)
    assert jnp.allclose(
        L @ L.T, L_fullrank @ L_fullrank.T, atol=1e-5
    ), "Cholesky factorizations do not match"

    # Test Christoffel symbols
    chris_full = fullrank_model.christoffel_fn(x, v)
    chris_monge = monge_model.christoffel_fn(x, v)
    assert jnp.allclose(
        chris_full, chris_monge, atol=1e-5
    ), f"Christoffel symbols do not match {chris_full} != {chris_monge}"

    # Test geodesic_fn
    t = 1.0
    x_out_fullrank, v_out_fullrank = fullrank_model.geodesic_fn(x, v, t)
    x_out_monge, v_out_monge = monge_model.geodesic_fn(x, v, t)
    assert jnp.allclose(
        x_out_fullrank, x_out_monge, atol=1e-5
    ), f"geodesic_fn outputs (x) do not match {x_out_fullrank} != {x_out_monge}"
    assert jnp.allclose(
        v_out_fullrank, v_out_monge, atol=1e-5
    ), f"geodesic_fn outputs (v) do not match {v_out_fullrank} != {v_out_monge}"

    print("All tests passed.")


# Run the test
test_inverse_monge_classes()
