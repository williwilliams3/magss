import jax
import jax.numpy as jnp

from .distribution import Distribution


class PhiFour(Distribution):
    def __init__(self, dim, a=0.1, beta=20.0):
        self.a = a
        self.beta = beta
        self.dim = dim
        self.log_Z = 0.0
        self.n_plots = 0
        self.can_sample = False

    def V(self, x):
        coef = self.a * self.dim
        diffs = 1.0 - jnp.square(x)
        V = jnp.dot(diffs, diffs) / 4 / coef
        return V

    def U(self, x):

        # Pad x with zeros on both sides
        x_ = jnp.pad(x, pad_width=1, mode="constant", constant_values=0)

        diffs = x_[1:] - x_[:-1]
        grad_term = jnp.dot(diffs, diffs) / 2
        coef = self.a * self.dim
        return grad_term * coef

    def loglik(self, x):
        return -self.beta * (self.U(x) + self.V(x))

    def logprior(self, x):
        return 0.0

    def logdensity_fn(self, x):
        return self.loglik(x) + self.logprior(x)
