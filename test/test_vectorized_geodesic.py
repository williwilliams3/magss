import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.random as jr
import jax.numpy as jnp
from metrics import Monge
from metrics.models import TwoGaussians
from metrics.geodesics import (
    numerical_integrate_geodesic_fn,
    numerical_integrate_geodesic_times_fn,
)
import time


dim = 1
alpha2 = 1.0
rng_key = jr.PRNGKey(42)
x = jnp.ones((dim,)) * -1.5 + 0.1 * jr.normal(rng_key, (dim,))
v = jnp.ones((dim,))

logdensity_fn = TwoGaussians(dim).logdensity_fn

# Initialize both classes
metric = Monge(dim, logdensity_fn, alpha2)


def geodesic_single_fn(x, v, t):
    dim = metric.dim
    output = numerical_integrate_geodesic_fn(dim, metric.christoffel_fn, x, v, t)
    x_new = output[-1, :dim]
    v_new = output[-1, dim:]
    return x_new, v_new


def geodesic_multiple_fn(x, v, ts):
    dim = metric.dim
    output = numerical_integrate_geodesic_times_fn(dim, metric.christoffel_fn, x, v, ts)
    x_new = output[:, :dim]
    v_new = output[:, dim:]
    return x_new, v_new


for t in [1.0, -1.0]:
    # ts = jnp.linspace(0.0, t, 10)
    ts = jnp.array([t])
    start_time = time.time()
    x_out_monge1, v_out_monge1 = jax.vmap(geodesic_single_fn, in_axes=(None, None, 0))(
        x, v, ts
    )
    jax.block_until_ready(x_out_monge1)
    time_elapsed1 = time.time() - start_time
    start_time = time.time()
    x_out_monge2, v_out_monge2 = geodesic_multiple_fn(x, v, ts)
    jax.block_until_ready(x_out_monge2)
    time_elapsed2 = time.time() - start_time
    print("times", ts)
    print("shapes", x_out_monge1.shape, x_out_monge2.shape)
    print(f"Elapsed time (single): {time_elapsed1}")
    print(f"Elapsed time (multiple): {time_elapsed2}")
    # Assert values are the same
    assert jnp.allclose(x_out_monge1, x_out_monge2), f"{x_out_monge1} != {x_out_monge2}"
    assert jnp.allclose(
        v_out_monge1, v_out_monge2, rtol=1e-4
    ), f"{v_out_monge1-v_out_monge2} != 0"
print("All tests passed!")
