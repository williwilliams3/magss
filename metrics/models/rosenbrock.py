import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from .distribution import Distribution


class Rosenbrock(Distribution):

    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Rosenbrock"
        self.a = 1.0
        self.b = 100.0
        self.dim = dim
        self.xlim = [-2, 3]
        self.ylim = [-1, 10]
        self.true_dist_levels = [-5.0, -3.0, -1.5]

    def logdensity_fn(self, theta):
        b = self.b
        a = self.a
        theta = jnp.array(theta)
        # First term
        logpdf = -((theta[0] - a) ** 2)
        # Terms dependent on previous term
        logpdf -= b * jnp.sum((theta[1:] - theta[:-1] ** 2) ** 2)
        return logpdf

    def transform_fn(self, z):
        dim = self.dim
        a = self.a
        b = self.b
        if z.ndim == 1:
            samples = []
            theta_1 = 1 / jnp.sqrt(2) * z[0] + a
            samples.append(theta_1)
            for i in range(1, dim):
                theta_i = 1 / jnp.sqrt(2 * b) * z[i] + samples[-1] ** 2
                samples.append(theta_i)
            return jnp.array(samples)
        samples = []
        theta_1 = 1 / jnp.sqrt(2) * z[:, 0] + a
        samples.append(theta_1[:, None])
        for i in range(1, dim):
            theta_i = 1 / jnp.sqrt(2 * b) * z[:, i] + samples[-1].flatten() ** 2
            samples.append(theta_i[:, None])
        return jnp.concatenate(samples, axis=1)

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        Z = jr.normal(rng_key, shape=(num_samples, dim))
        return self.transform_fn(Z)

    def inverse_jacobian_fn(self, theta):
        D = self.dim
        b = self.b
        first_entry = jnp.sqrt(2)
        rest_entries = jnp.sqrt(2 * b) * jnp.ones(D - 1)
        diag_term = jnp.concatenate([jnp.array([first_entry]), rest_entries])
        low_diag_term = -2 * jnp.sqrt(2 * b) * theta[:-1]
        result_matrix = jnp.diag(diag_term) + jnp.diag(low_diag_term, k=-1)
        return result_matrix

    def jacobian_fn(self, z):
        D = self.dim
        a = self.a
        b = self.b

        first_entry = 1 / jnp.sqrt(2)
        rest_entries = 1 / jnp.sqrt(2 * b) * jnp.ones(D - 1)
        diag_term = jnp.concatenate([jnp.array([first_entry]), rest_entries])
        low_diag_term = jnp.sqrt(2) * a + z[:-1]
        result_matrix = jnp.diag(diag_term) + jnp.diag(low_diag_term, k=-1)
        return result_matrix

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)
