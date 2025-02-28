import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jss
from typing import NamedTuple, Callable
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from blackjax.base import SamplingAlgorithm


logdensity_fn = lambda x: jss.norm.logpdf(x, loc=0.0, scale=1.0).sum()


class SliceState(NamedTuple):
    position: ArrayTree
    logdensity: float


class SliceInfo(NamedTuple):
    logthreshold: float
    boundaries: ArrayLikeTree


def init(position: ArrayLikeTree):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity)


def build_shrinkage_kernel():

    def step_out(logthreshold):
        # Computes exact interval for isotropic gaussian
        r = jnp.sqrt(-jnp.log(2 * jnp.pi) - 2 * logthreshold)
        l = -r
        return (l, r)

    def shrinkage(rng_key, x, boundaries):

        (l, r) = boundaries
        proposal = jax.random.uniform(rng_key, minval=l, maxval=r, shape=x.shape)
        logdensityproposal = logdensity_fn(proposal)
        return SliceState(proposal, logdensityproposal)

    def kernel(rng_key: PRNGKey, state: SliceState):
        x, logdensity = state
        rng_key1, rng_key2 = jax.random.split(rng_key)
        logthreshold = logdensity + jnp.log(jax.random.uniform(rng_key1))
        boundaries = step_out(logthreshold)
        proposal = shrinkage(rng_key2, x, boundaries)

        return proposal, SliceInfo(logthreshold, boundaries)

    return kernel


class univariate_slice_sampler_idealiced:

    init = staticmethod(init)
    build_kernel = staticmethod(build_shrinkage_kernel)

    def __new__(
        cls,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: ArrayLikeTree, rng_key=None):
            del rng_key
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state)

        return SamplingAlgorithm(init_fn, step_fn)
