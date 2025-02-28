import jax.random as jr
import jax.numpy as jnp
from .metrics import Metric


class Euclidean(Metric):

    def __init__(self, dim):
        super().__init__(dim)

    def sample_unit_ball(self, rng_key, x):
        v = self.sample_velocity(rng_key, x)
        v /= jnp.linalg.norm(v)
        return v

    def sample_velocity(self, rng_key, x):
        return jr.normal(rng_key, x.shape)

    def geodesic_fn(self, x, v, t):
        position = x + t * v
        velocity = v
        return position, velocity

    def geodesic_vectorized_fn(self, x, v, ts):
        positions = x + jnp.outer(ts, v)
        velocities = jnp.repeat(v[None, :], len(ts), axis=0)
        return positions, velocities

    def squared_norm_fn(x, v):
        return jnp.square(v).sum()

    def log_determinant_fn(self, x):
        return 0.0
