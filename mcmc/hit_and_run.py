import jax
import jax.numpy as jnp
from typing import Callable
from blackjax.types import ArrayLikeTree, PRNGKey
from blackjax.base import SamplingAlgorithm
from .univariate_slice_sampler import (
    step_out,
    shrinkage,
    no_shrinkage,
    SliceState,
    SliceInfo,
)


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity)


def build_shrinkage_kernel(w: float, m: int, do_shrinkage: bool = True):

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logdensity_fn: Callable,
    ):
        x, logdensity = state
        rng_key1, rng_key2, rng_key3, rng_key4 = jax.random.split(rng_key, 4)

        # Sample a random direction of unit length
        v = jax.random.normal(rng_key1, x.shape)
        v /= jnp.linalg.norm(v)

        target_fn = lambda theta: (x + theta * v, logdensity_fn(x + theta * v))
        logthreshold = logdensity + jnp.log(jax.random.uniform(rng_key2))
        boundaries, num_reject1 = step_out(rng_key3, target_fn, logthreshold, w, m)
        if do_shrinkage:
            proposal, (num_reject2, thetas) = shrinkage(
                rng_key4, target_fn, logthreshold, boundaries
            )
        else:
            proposal, (num_reject2, thetas) = no_shrinkage(
                rng_key4, target_fn, logthreshold, boundaries
            )

        return proposal, SliceInfo(
            num_reject1, num_reject2, logthreshold, boundaries, thetas
        )

    return kernel


class hit_and_run:

    init = staticmethod(init)
    build_kernel = staticmethod(build_shrinkage_kernel)

    def __new__(
        cls,
        logdensity_fn: Callable,
        step_size: float,
        max_stepouts: int,
        do_shrinkage: bool = False,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(
            w=step_size, m=max_stepouts, do_shrinkage=do_shrinkage
        )

        def init_fn(position: ArrayLikeTree, rng_key=None):
            del rng_key
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn)

        return SamplingAlgorithm(init_fn, step_fn)
