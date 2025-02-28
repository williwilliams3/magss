import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from .distribution import Distribution


class Funnel(Distribution):

    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Funnel"
        self.mean = 0.0
        self.sigma = 3.0
        self.xlim = [-10.0, 10.0]
        self.ylim = [-10.0, 10.0]
        self.true_dist_levels = [-10.0, -5.0, -2.0]

    def logdensity_fn(self, x):
        mean = self.mean
        sigma = self.sigma
        dim = self.dim
        return jss.norm.logpdf(x[dim - 1], loc=0.0, scale=sigma) + jnp.sum(
            jss.norm.logpdf(x[: dim - 1], loc=mean, scale=jnp.exp(0.5 * x[dim - 1]))
        )

    def transform_fn(self, z):
        dim = self.dim
        sigma = self.sigma
        mean = self.mean

        if z.ndim == 1:
            theta_D = sigma * z[-1]
            z_D1 = z[0 : (dim - 1)] * jnp.exp(0.5 * theta_D) + mean
            return jnp.append(z_D1, theta_D)

        theta_D = sigma * z[:, -1]
        theta_D1 = z[:, 0 : (dim - 1)] * jnp.exp(0.5 * theta_D[:, None]) + mean

        return jnp.c_[theta_D1, theta_D]

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        Z = jr.normal(rng_key, shape=(num_samples, dim))
        return self.transform_fn(Z)

    def inverse_jacobian_fn(self, theta):
        dim = self.dim
        upper_rows = jnp.c_[
            jnp.exp(-0.5 * theta[-1]) * jnp.eye(dim - 1),
            -0.5 * jnp.exp(-0.5 * theta[-1]) * theta[0 : (dim - 1)],
        ]
        lowest_row = jnp.append(jnp.zeros(dim - 1), 1.0 / self.sigma)
        inverse_jacobian = jnp.r_[upper_rows, [lowest_row]]
        return inverse_jacobian

    def jacobian_fn(self, z):
        dim = self.dim
        sigma = self.sigma
        exp_term = jnp.exp(0.5 * sigma * z[-1])
        upper_rows = jnp.c_[
            exp_term * jnp.eye(dim - 1), 0.5 * sigma * exp_term * z[0 : (dim - 1)]
        ]
        lowest_row = jnp.append(jnp.zeros(dim - 1), sigma)
        jacobian = jnp.r_[upper_rows, [lowest_row]]
        return jacobian

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)
