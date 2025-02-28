import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
from typing import NamedTuple
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from metrics.metrics import Metric


class State(NamedTuple):
    position: ArrayTree


class Info(NamedTuple):
    velocity: ArrayTree


def init(position):
    return State(position)


def build_kernel(metric_class):

    def kernel(rng_key, state):

        x = state.position
        v = metric_class.sample_velocity(rng_key, x)
        sample, velocity = metric_class.geodesic_fn(x, v, t=1.0)

        return State(sample), Info(velocity)

    return kernel


class riemannianlaplace:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(
        cls,
        metric: Metric,
    ) -> SamplingAlgorithm:

        kernel = cls.build_kernel(metric)

        def init_fn(position: ArrayLikeTree, rng_key):
            # Position at MAP
            del rng_key
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state)

        return SamplingAlgorithm(init_fn, step_fn)
