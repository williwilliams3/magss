import jax
import jax.numpy as jnp
import jax.random as jr
from .distribution import Distribution


class HybridRosenbrock(Distribution):
    # refer to: https://onlinelibrary.wiley.com/doi/full/10.1111/sjos.12532
    def __init__(self, dim=2):
        self.a = 1.0
        self.b = 100.0
        self.n1 = 3  # size of blocks
        self.dim = dim
        self.name = "Rosenbrock"
        self.xlim = [-2, 3]
        self.ylim = [-1, 10]
        self.true_dist_levels = [-5.0, -3.0, -1.5]

    def logdensity_fn(self, theta):
        D = self.dim
        n1 = self.n1
        b = self.b
        a = self.a
        theta = jnp.array(theta)
        # First term
        logpdf = -((theta[0] - a) ** 2)
        # Terms dependent on x[0]
        step_indices = jnp.arange(1, D, n1)
        logpdf -= b * jnp.sum((theta[step_indices] - theta[0] ** 2) ** 2)
        # Terms dependent on x[i-1]
        # remove terms dependent on x[0]
        indices = jnp.arange(1, D)
        mask = jnp.mod(indices, n1) != 0
        middle_elements = indices[jnp.nonzero(mask, size=D - 1)[0]]
        logpdf -= b * jnp.sum(
            (theta[middle_elements] - theta[middle_elements - 1] ** 2) ** 2
        )
        return logpdf

    def inverse_jacobian_fn(self, theta):
        D = self.dim
        b = self.b
        n1 = self.n1
        first_entry = jnp.sqrt(2)
        rest_entries = jnp.sqrt(2 * b) * jnp.ones(D - 1)
        diag_term = jnp.concatenate([jnp.array([first_entry]), rest_entries])
        low_diag_term = -2 * jnp.sqrt(2 * b) * theta[:-1].at[0::n1].set(0.0)
        result_matrix = jnp.diag(diag_term) + jnp.diag(low_diag_term, k=-1)
        result_matrix_new = result_matrix.at[1::n1, 0].add(
            -2 * jnp.sqrt(2 * b) * theta[0]
        )
        return result_matrix_new

    def jacobian_fn(self, z):
        return jax.jacfwd(self.transform_fn)(z)

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        Z = jr.normal(rng_key, shape=(num_samples, dim))
        return self.transform_fn(Z)

    def transform_fn(self, z):
        D = self.dim
        a = self.a
        b = self.b
        n1 = self.n1
        samples = []
        if z.ndim == 1:
            theta_1 = 1 / jnp.sqrt(2) * z[0] + a
            samples.append(theta_1)
            for i in range(1, D):
                if (i - 1) % n1 == 0:
                    index_dependency = 0
                else:
                    index_dependency = -1
                theta_i = (
                    1 / jnp.sqrt(2 * b) * z[i] + samples[index_dependency] ** 2
                ).squeeze()
                samples.append(theta_i)
            return jnp.array(samples)
        theta_1 = 1 / jnp.sqrt(2) * z[:, 0] + a
        samples.append(theta_1[:, None])
        for i in range(1, D):
            if (i - 1) % n1 == 0:
                index_dependency = 0
            else:
                index_dependency = -1
            theta_i = (
                1 / jnp.sqrt(2 * b) * z[:, i] + samples[index_dependency].flatten() ** 2
            )
            samples.append(theta_i[:, None])
        return jnp.concatenate(samples, axis=1)
