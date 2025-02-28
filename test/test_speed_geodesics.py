import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.random as jr
import jax.numpy as jnp
from metrics import InverseMonge, Monge
from metrics.models import TwoGaussians
import time


dim = 1
alpha2 = 1.0
rng_key = jr.PRNGKey(42)
x = jnp.ones((dim,)) * -1.5 + 0.1 * jr.normal(rng_key, (dim,))
v = jnp.ones((dim,))

logdensity_fn = TwoGaussians(dim).logdensity_fn

# Initialize both classes
metric1 = Monge(dim, logdensity_fn, alpha2)
metric2 = InverseMonge(dim, logdensity_fn, alpha2, use_univariate=False)

# geodesic_fn1 = jax.jit(metric1.geodesic_fn)
# geodesic_fn2 = jax.jit(metric2.geodesic_fn)

geodesic_fn1 = metric1.geodesic_fn
geodesic_fn2 = metric2.geodesic_fn


# t = jnp.arange(8, dtype=jnp.float32)
t = 1.0
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

print(f"Elapsed time (Monge): {elapsed_time1}")
print(f"Elapsed time (Inverse Monge): {elapsed_time2}")

print(x_out_monge1.shape)
print(x_out_monge2.shape)
