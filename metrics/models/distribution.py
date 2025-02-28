import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss


class Distribution:

    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Distribution"

    def logdensity_fn(self, x):
        raise NotImplementedError

    def sample(self, rng_key, num_samples=1):
        raise NotImplementedError

    def fisher_metric_fn(self, x):
        raise NotImplementedError

    def inverse_jacobian_fn(self, x):
        raise NotImplementedError

    def jacobian_fn(self, z):
        raise NotImplementedError
