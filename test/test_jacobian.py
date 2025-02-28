import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.random as jr
import jax.numpy as jnp
from metrics.models import Rosenbrock, Funnel, Squiggle


if __name__ == "__main__":
    rng_key = jr.key(0)
    dim = 8

    for dist in [Funnel(dim), Squiggle(dim), Rosenbrock(dim)]:
        print(f"Testing {dist.name} distribution")
        z = jr.normal(rng_key, dim)
        x = dist.transform_fn(z)

        assert jnp.allclose(
            dist.jacobian_fn(z) @ dist.inverse_jacobian_fn(x),
            jnp.eye(dim),
            atol=1e-5,
            rtol=1e-5,
        ), f"Assertion failed: \nComputed: {dist.jacobian_fn(z) @ dist.inverse_jacobian_fn(x)} \nExpected: {jnp.eye(dim)}"
        assert jnp.allclose(
            dist.inverse_jacobian_fn(x) @ dist.jacobian_fn(z),
            jnp.eye(dim),
            atol=1e-4,
            rtol=1e-4,
        ), f"Assertion failed: \nComputed: { dist.inverse_jacobian_fn(x) @ dist.jacobian_fn(z) } \nExpected: {jnp.eye(dim)}"

    print("Test passed!")
