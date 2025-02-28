import jax
import jax.numpy as jnp
import jax.scipy.stats as jss


class Gaussian:

    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Gaussian"
        self.xlim = [-3.0, 3.0]
        self.ylim = [-3.0, 3.0]
        self.true_dist_levels = [-6.3378773, -3.8378773, -2.337877]

    def logdensity_fn(self, x):
        return jss.norm.logpdf(x).sum()

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        Z = jax.random.normal(rng_key, shape=(num_samples, dim))
        return Z

    def fisher_metric_fn(self, theta):
        return jnp.eye(self.dim)

    def jacobian_fn(self, theta):
        return jnp.eye(self.dim)

    def inverse_jacobian_fn(self, theta):
        return jnp.eye(self.dim)
