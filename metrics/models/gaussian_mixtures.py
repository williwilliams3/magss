import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy.stats as jss


default_weights = [0.8, 0.2]


class TwoGaussians:

    def __init__(self, dim, weights=default_weights):
        self.dim = dim
        self.name = "TwoGaussians"
        self.means = jnp.array(
            [
                jnp.ones(dim) * -1,
                jnp.ones(dim) * 1,
            ]
        )
        self.sigma = jnp.array([0.1])
        self.weights = jnp.array(weights)
        self.xlim = [-1.5, 1.5]
        self.ylim = [-1.5, 1.5]
        self.true_dist_levels = jnp.array([-6.3378773, -3.8378773, -2.337877])

    def logdensity_fn(self, x):
        means = self.means
        sigma = self.sigma
        weights = self.weights

        logdensity_fn = logdensity_gaussian_mixture(means, sigma, weights)
        return logdensity_fn(x)

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        means = self.means
        sigma = self.sigma
        weights = self.weights
        sample_fn = sample_gaussian_mixture(dim, means, sigma, weights)
        return sample_fn(rng_key, num_samples)


class NineGaussians:

    def __init__(self, dim):
        assert dim == 2
        self.dim = dim
        self.name = "NineGaussians"
        self.means = jnp.array(
            [(i, j) for i in range(-3, 4, 3) for j in range(-3, 4, 3)]
        )
        self.sigma = 0.1
        self.weights = jnp.ones(len(self.means)) / len(self.means)
        self.xlim = [-4.5, 4.5]
        self.ylim = [-4.5, 4.5]

    def logdensity_fn(self, x):
        means = self.means
        sigma = self.sigma
        weights = self.weights
        logdensity_fn = logdensity_gaussian_mixture(means, sigma, weights)
        return logdensity_fn(x)

    def sample(self, rng_key, num_samples=1):
        dim = self.dim
        means = self.means
        sigma = self.sigma
        weights = self.weights
        sample_fn = sample_gaussian_mixture(dim, means, sigma, weights)
        return sample_fn(rng_key, num_samples)


class GMM:

    def __init__(
        self, rng_key, dim, means, log_sigma, num_mixtures=None, loc_scale=40.0
    ):
        key = rng_key
        self.dim = dim
        self.name = "GMM"
        self.num_mixtures = num_mixtures
        if means is not None:
            self.means = means
        else:
            assert self.num_mixtures is not None and loc_scale is not None
            self.means = (
                (jax.random.uniform(key=key, shape=(self.num_mixtures, self.dim)) - 0.5)
                * 2.0
                * loc_scale
            )
        self.sigma = jax.nn.softplus(log_sigma)
        self.weights = jnp.ones(len(self.means)) / len(self.means)
        self.xlim = [
            jnp.min(self.means[:, 0]) - 5.0 * self.sigma,
            jnp.max(self.means[:, 0]) + 5.0 * self.sigma,
        ]
        self.ylim = [
            jnp.min(self.means[:, 1]) - 5.0 * self.sigma,
            jnp.max(self.means[:, 1]) + 5.0 * self.sigma,
        ]

    def logdensity_fn(self, x):
        means = self.means
        sigma = self.sigma
        weights = self.weights
        logdensity_fn = logdensity_gaussian_mixture(means, sigma, weights)
        return logdensity_fn(x)

    def sample(self, rng_key, num_samples=10000):
        dim = self.dim
        means = self.means
        sigma = self.sigma
        weights = self.weights
        sample_fn = sample_gaussian_mixture(dim, means, sigma, weights)
        return sample_fn(rng_key, num_samples)


def logdensity_gaussian_mixture(means, sigma, weights):
    def logdensity_fn(x):
        gaussian_densities = jnp.asarray(
            [jss.norm.logpdf(x, loc=mu, scale=sigma).sum() for mu in means]
        )
        return jax.scipy.special.logsumexp(a=gaussian_densities, b=weights)

    return logdensity_fn


def sample_gaussian_mixture(dim, means, sigma, weights):
    def sample_fn(rng_key, N=1):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        component_indices = jr.choice(rng_key1, a=len(weights), shape=(N,), p=weights)
        Z = jr.normal(rng_key2, shape=(N, dim))
        X = means[component_indices] + sigma * Z
        return X

    return sample_fn
