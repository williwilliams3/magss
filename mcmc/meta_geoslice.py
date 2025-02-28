import jax
import jax.random as jr
from blackjax.base import SamplingAlgorithm
from typing import Callable, NamedTuple
from blackjax.types import ArrayLikeTree, PRNGKey

from .univariate_slice_sampler import (
    SliceState,
    SliceInfo,
)
from .geoslice_vectorized import geodesic_slice_sampler as _meta_sampler
from metrics.metrics import Metric


class MetaInfo(NamedTuple):
    info_meta: SliceInfo
    info_sba: NamedTuple
    states_meta: ArrayLikeTree
    states_sba: ArrayLikeTree


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity)


def build_kernel(
    logdensity_fn: Callable,
    step_size: float,
    max_stepouts: int,
    metric: Metric,
    max_shirnkage: int,
):
    meta_sampler = _meta_sampler(
        logdensity_fn,
        step_size,
        max_stepouts,
        metric,
        max_shrinkage=max_shirnkage,
    )

    def kernel(
        rng_key: PRNGKey,
        meta_state: SliceState,
        sampler_fn: Callable,
        alg_steps: int,
        sweeps: int = 1,
    ):
        def body_fn(rng_key, state):

            rng_key, rng_key1, rng_key2, rng_key3 = jr.split(rng_key, 4)

            # Sweep Approx Geodesic Slice Sampler
            keys_agss = jr.split(rng_key1, sweeps)

            def one_step_agss(state, key):
                state, info = meta_sampler.step(key, state)
                return state, (state, info)

            state_meta, (states_meta, info_meta) = jax.lax.scan(
                one_step_agss, state, keys_agss
            )
            # state, info = meta_sampler.step(rng_key1, state)

            # Score based MCMC
            x_init = state_meta.position
            sampling_alg = sampler_fn(logdensity_fn)
            state_sba = sampling_alg.init(position=x_init, rng_key=rng_key2)
            keys_sba = jr.split(rng_key3, alg_steps)

            def one_step_sba(state, rng_key):
                state, info = sampling_alg.step(rng_key, state)
                return state, (state, info)

            state_sba, (states_sba, info_sba) = jax.lax.scan(
                one_step_sba, state_sba, keys_sba
            )

            proposal = state_sba.position
            logdensity = logdensity_fn(state_sba.position)
            return SliceState(proposal, logdensity), MetaInfo(
                info_meta, info_sba, states_meta, states_sba
            )

        return body_fn(rng_key, meta_state)

    return kernel


class meta_geodesic_slice_sampler:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(
        cls,
        logdensity_fn: Callable,
        sampler_fn: Callable,
        alg_steps: int,
        step_size: float,
        max_stepouts: int,
        metric: Metric,
        max_shirnkage: int = 100,
    ) -> SamplingAlgorithm:

        kernel = cls.build_kernel(
            logdensity_fn, step_size, max_stepouts, metric, max_shirnkage
        )

        def init_fn(position: ArrayLikeTree, rng_key: PRNGKey):
            del rng_key
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state: SliceState):
            return kernel(rng_key, state, sampler_fn, alg_steps)

        return SamplingAlgorithm(init_fn, step_fn)
