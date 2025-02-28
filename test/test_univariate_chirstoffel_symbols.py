import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.random as jr
import jax.numpy as jnp
from metrics import InverseMonge
from metrics.models import TwoGaussians
import time


def test_inverse_monge_classes():
    dim = 1
    alpha2 = 1.0
    rng_key = jr.PRNGKey(42)
    x = jnp.ones((dim,)) + 0.1 * jr.normal(rng_key, (dim,))

    # Dummy log-density function
    logdensity_fn = TwoGaussians(dim).logdensity_fn
    # logdensity_fn = lambda x: -jnp.sum(x**2)

    # Initialize both classes
    monge_model1 = InverseMonge(dim, logdensity_fn, alpha2, use_univariate=True)
    monge_model2 = InverseMonge(dim, logdensity_fn, alpha2, use_univariate=False)

    # Test squared_norm_fn
    v = jr.normal(rng_key, x.shape)

    # Test Christoffel symbols
    chris_monge1 = monge_model1.christoffel_fn(x, v)
    chris_monge2 = monge_model2.christoffel_fn(x, v)
    assert jnp.allclose(
        chris_monge1, chris_monge2, atol=1e-4
    ), f"Christoffel symbols do not match {chris_monge1} != {chris_monge2}"

    # Test geodesic_fn
    t = 1.0
    geodesic_fn1 = jax.jit(monge_model1.geodesic_fn)
    geodesic_fn2 = jax.jit(monge_model2.geodesic_fn)

    start_time = time.time()
    x_out_monge1, v_out_monge1 = geodesic_fn1(x, v, t)
    jax.block_until_ready(x_out_monge1)
    jax.block_until_ready(v_out_monge1)
    elapsed_time1 = time.time() - start_time
    start_time = time.time()
    x_out_monge2, v_out_monge2 = geodesic_fn2(x, v, t)
    jax.block_until_ready(x_out_monge1)
    jax.block_until_ready(v_out_monge1)
    elapsed_time2 = time.time() - start_time

    print(f"Elapsed time (univariate): {elapsed_time1}")
    print(f"Elapsed time (multivariate): {elapsed_time2}")

    assert jnp.allclose(
        x_out_monge1, x_out_monge2, atol=1e-4
    ), f"geodesic_fn outputs (x) do not match {x_out_monge1} != {x_out_monge2}"
    assert jnp.allclose(
        v_out_monge1, v_out_monge2, atol=1e-4
    ), f"geodesic_fn outputs (v) do not match {v_out_monge1} != {v_out_monge2}"

    print("All tests passed.")


# Run the test
test_inverse_monge_classes()
