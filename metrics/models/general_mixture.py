import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy.stats as jss
from .distribution import Distribution


class GeneralFunnel(Distribution):
    # Funnel distribution with arbitrary location and scale parameters

    def __init__(self, dim=2, loc=jnp.zeros(2), scale=jnp.ones(2)):
        self.dim = dim
        self.name = "Funnel"
        self.loc = loc
        self.scale = scale

        self.sigma = 3.0
        self.xlim = [-10.0, 10.0]
        self.ylim = [-10.0, 10.0]
        self.true_dist_levels = [-10.0, -5.0, -2.0]

    def logdensity_fn(self, x):
        loc = self.loc
        scale = self.scale
        y = (x - loc) / scale
        z = self.inverse_transform_fn(y)
        logdet1 = jnp.sum(-jnp.log(scale))
        logdet2 = self.logdeterminant_inverse_fn(x)
        return jss.norm.logpdf(z).sum() + logdet1 + logdet2

    def transform_fn(self, z):
        dim = self.dim
        sigma = self.sigma

        if z.ndim == 1:
            theta_D = sigma * z[-1]
            z_D1 = z[0 : (dim - 1)] * jnp.exp(0.5 * theta_D)
            return jnp.append(z_D1, theta_D)

        theta_D = sigma * z[:, -1]
        theta_D1 = z[:, 0 : (dim - 1)] * jnp.exp(0.5 * theta_D[:, None])

        return jnp.c_[theta_D1, theta_D]

    def inverse_transform_fn(self, x):
        dim = self.dim
        sigma = self.sigma

        if x.ndim == 1:
            theta_D = x[-1]
            z_D1 = (x[0 : (dim - 1)]) * jnp.exp(-0.5 * theta_D)
            z_D = theta_D / sigma
            return jnp.append(z_D1, z_D)

        theta_D = x[:, -1]
        z_D1 = (x[:, 0 : (dim - 1)]) * jnp.exp(-0.5 * theta_D[:, None])
        z_D = theta_D / sigma

        return jnp.c_[z_D1, z_D]

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        loc = self.loc
        scale = self.scale
        Z = jr.normal(rng_key, shape=(num_samples, dim))
        Y = self.transform_fn(Z)
        X = Y * scale + loc
        return X

    def logdeterminant_inverse_fn(self, x):
        sigma = self.sigma
        dim = self.dim

        if x.ndim == 1:
            x_D = x[-1]  # Last element in 1D case
            det = jnp.exp(-(dim - 1) * x_D / 2) / sigma
        else:
            x_D = x[:, -1]  # Last column in 2D (batch) case
            det = jnp.exp(-(dim - 1) * x_D / 2) / sigma

        return det

    def inverse_jacobian_fn(self, theta):
        dim = self.dim
        upper_rows = jnp.c_[
            jnp.exp(-0.5 * theta[-1]) * jnp.eye(dim - 1),
            -0.5 * jnp.exp(-0.5 * theta[-1]) * theta[0 : (dim - 1)],
        ]
        lowest_row = jnp.append(jnp.zeros(dim - 1), 1.0 / self.sigma)
        inverse_jacobian = jnp.r_[upper_rows, [lowest_row]]
        return inverse_jacobian

    def fisher_metric_fn(self, theta):
        Sigma_inv_sqrt = jnp.diag(1 / self.scale)
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = (
            Sigma_inv_sqrt.T @ inverse_jacobian.T @ inverse_jacobian @ Sigma_inv_sqrt
        )
        return 0.5 * (metric + metric.T)


class GeneralRosenbrock(Distribution):

    def __init__(self, dim=2, loc=jnp.ones(2), scale=jnp.ones(2)):
        self.dim = dim
        self.name = "Rosenbrock"
        self.loc = loc
        self.scale = scale
        self.a = 1.0
        self.b = 100.0
        self.dim = dim
        self.xlim = [-2, 3]
        self.ylim = [-1, 10]
        self.true_dist_levels = [-5.0, -3.0, -1.5]

    def logdensity_fn(self, x):
        loc = self.loc
        scale = self.scale
        y = (x - loc) / scale
        z = self.inverse_transform_fn(y)
        logdet1 = jnp.sum(-jnp.log(scale))
        logdet2 = self.logdeterminant_inverse_fn(x)
        return jss.norm.logpdf(z).sum() + logdet1 + logdet2

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

    def inverse_transform_fn(self, x):
        dim = self.dim
        a = self.a
        b = self.b

        if x.ndim == 1:
            samples = []
            z_1 = jnp.sqrt(2) * (x[0] - a)
            samples.append(z_1)
            for i in range(1, dim):
                z_i = jnp.sqrt(2 * b) * (x[i] - x[i - 1] ** 2)
                samples.append(z_i)

            return jnp.array(samples)

        samples = []
        z_1 = jnp.sqrt(2) * (x[:, 0] - a)
        samples.append(z_1[:, None])
        for i in range(1, dim):
            z_i = jnp.sqrt(2 * b) * (x[:, i] - x[:, i - 1] ** 2)
            samples.append(z_i[:, None])

        return jnp.concatenate(samples, axis=1)

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        loc = self.loc
        scale = self.scale
        Z = jr.normal(rng_key, shape=(num_samples, dim))
        return self.transform_fn(Z) * scale + loc

    def inverse_jacobian_fn(self, theta):
        D = self.dim
        b = self.b
        first_entry = jnp.sqrt(2)
        rest_entries = jnp.sqrt(2 * b) * jnp.ones(D - 1)
        diag_term = jnp.concatenate([jnp.array([first_entry]), rest_entries])
        low_diag_term = -2 * jnp.sqrt(2 * b) * theta[:-1]
        result_matrix = jnp.diag(diag_term) + jnp.diag(low_diag_term, k=-1)
        return result_matrix

    def logdeterminant_inverse_fn(self, theta):
        b = self.b
        dim = self.dim
        logdet = 0.5 * jnp.log(2) + 0.5 * (dim - 1) * jnp.log(2 * b)
        return logdet

    def fisher_metric_fn(self, theta):
        Sigma_inv_sqrt = jnp.diag(1 / self.scale)
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = (
            Sigma_inv_sqrt.T @ inverse_jacobian.T @ inverse_jacobian @ Sigma_inv_sqrt
        )
        return 0.5 * (metric + metric.T)


class GeneralSquiggle(Distribution):

    def __init__(self, dim=2, loc=jnp.zeros(2), scale=jnp.ones(2)):
        self.dim = dim
        self.name = "Squiggle"
        self.a = 1.5
        self.mean = jnp.zeros(dim)
        self.diagonal = jnp.array([5] + (dim - 1) * [0.05])
        self.xlim = [-9, 9]
        self.ylim = [-3, 3]
        self.true_dist_levels = [-5.0, -3.0, -1.5]
        self.loc = loc
        self.scale = scale

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

    def logdensity_fn(self, x):
        loc = self.loc
        scale = self.scale
        y = (x - loc) / scale
        z = self.inverse_transform_fn(y)
        logdet1 = jnp.sum(-jnp.log(scale))

        return jss.norm.logpdf(z).sum() + logdet1

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

    def inverse_transform_fn(self, x):
        a = self.a
        mean = self.mean
        P = self.Precision_squareroot
        if x.ndim == 1:
            z = x
            z = z.at[1:].set(z[1:] + jnp.sin(a * x[0]))
            z = jnp.dot(P.T, z - mean)

            return z
        z = x
        z = z.at[:, 1:].set(z[:, 1:] + jnp.sin(a * x[:, 0])[:, None])
        z = jnp.dot(z - mean, P.T)
        return z

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        Z = jr.normal(rng_key, shape=(num_samples, dim), dtype=jnp.float32)
        Y = self.transform_fn(Z)
        X = Y * self.scale + self.loc
        return X

    def inverse_jacobian_fn(self, theta):
        D = self.dim
        a = self.a
        top_row = jnp.append(1.0, jnp.zeros(D - 1))
        bottom_rows = jnp.c_[
            a * jnp.cos(a * theta[0]) * jnp.ones(D - 1), jnp.eye(D - 1)
        ]
        inverse_jacobian = jnp.r_[[top_row], bottom_rows]
        return self.Precision_squareroot @ inverse_jacobian

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian_fn(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)


class GeneralMixture:

    def __init__(
        self,
        dim=2,
        distributionlist=[
            # GeneralFunnel(2, loc=jnp.array([0.0, 2.5]), scale=0.2 * jnp.ones(2)),
            GeneralSquiggle(2, loc=jnp.array([0.0, 0.0]), scale=0.2 * jnp.ones(2)),
            GeneralRosenbrock(2, loc=jnp.array([-2.5, 0.0]), scale=0.2 * jnp.ones(2)),
        ],
    ):
        self.dim = dim
        self.name = "TwoMixture"
        self.distributionlist = distributionlist
        self.weights = jnp.array([0.5, 0.5])
        self.xlim = [-3.0, 1.0]
        self.ylim = [-0.5, 2.0]
        self.true_dist_levels = jnp.array([-5.0, -3.0, -1.5]) * 0.2

    def logdensity_fn(self, x):
        weights = self.weights
        distributionlist = self.distributionlist

        logdensity_fn = logdensity_mixture(weights, distributionlist)
        return logdensity_fn(x)

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        weights = self.weights
        distributionlist = self.distributionlist
        sample_fn = sample_mixture(dim, weights, distributionlist)
        return sample_fn(rng_key, num_samples)


def logdensity_mixture(weights, distributionlist):
    def logdensity_fn(x):
        densities = jnp.asarray([dist.logdensity_fn(x) for dist in distributionlist])
        return jax.scipy.special.logsumexp(a=densities, b=weights)

    return logdensity_fn


def sample_mixture(dim, weights, distribution_list):
    def sample_fn(rng_key, N=1):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        component_indices = jr.choice(rng_key1, a=len(weights), shape=(N,), p=weights)
        X = jnp.zeros((N, dim))
        for i in range(len(weights)):
            num_samples = jnp.sum(component_indices == i)
            if num_samples > 0:
                samples = distribution_list[i].sample(rng_key2, num_samples=num_samples)
                X = X.at[component_indices == i].set(samples)
        return X

    return sample_fn
