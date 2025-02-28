import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from .distribution import Distribution


class Squiggle(Distribution):

    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Squiggle"
        self.a = 1.5
        self.mean = jnp.zeros(dim)
        self.diagonal = jnp.array([5] + (dim - 1) * [0.05])
        self.xlim = [-9, 9]
        self.ylim = [-3, 3]
        self.true_dist_levels = [-5.0, -3.0, -1.5]

    @property
    def Sigma(self):
        return jnp.diag(self.diagonal)

    @property
    def Covariance_squareroot(self):
        return jnp.diag(jnp.sqrt(self.diagonal))

    @property
    def Precision_squareroot(self):
        return jnp.diag(1 / jnp.sqrt(self.diagonal))

    def logdensity_fn(self, x):
        mean = self.mean
        Sigma = self.Sigma
        g = jnp.insert(x[1:] + jnp.sin(self.a * x[0]), 0, x[0])
        return jss.multivariate_normal.logpdf(g, mean, Sigma)

    def transform_fn(self, z):
        a = self.a
        mean = self.mean
        L = self.Covariance_squareroot
        if z.ndim == 1:
            theta = jnp.dot(L, z) + mean
            theta = theta.at[1:].set(theta[1:] - jnp.sin(a * theta[0]))
            return theta
        theta = z @ L + mean
        theta = theta.at[:, 1:].set(theta[:, 1:] - jnp.sin(a * theta[:, 0])[:, None])
        return theta

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        Z = jr.normal(rng_key, shape=(num_samples, dim), dtype=jnp.float32)
        return self.transform_fn(Z)

    def inverse_jacobian_fn(self, theta):
        D = self.dim
        a = self.a
        top_row = jnp.append(1.0, jnp.zeros(D - 1))
        bottom_rows = jnp.c_[
            a * jnp.cos(a * theta[0]) * jnp.ones(D - 1), jnp.eye(D - 1)
        ]
        inverse_jacobian = jnp.r_[[top_row], bottom_rows]
        return self.Precision_squareroot @ inverse_jacobian

    def jacobian_fn(self, z):
        # Assume z \sim N(0,I)
        dim = self.dim
        a = self.a
        z = self.Covariance_squareroot @ z + self.mean
        top_row = jnp.append(1.0, jnp.zeros(dim - 1))
        bottom_rows = jnp.c_[
            -a * jnp.cos(a * z[0]) * jnp.ones(dim - 1), jnp.eye(dim - 1)
        ]
        jacobian = jnp.r_[[top_row], bottom_rows]
        return jacobian @ self.Covariance_squareroot

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)
