import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Callable
from blackjax.types import ArrayLikeTree, PRNGKey

from blackjax.base import SamplingAlgorithm
from metrics.metrics import Metric
from .univariate_slice_sampler import (
    SliceState,
    SliceInfo,
)


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity)


def build_shrinkage_kernel(
    w: float,
    m: int,
    metric: Metric,
    max_shrinkage: int = 100,
):

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logdensity_fn: Callable,
    ):
        x, logdensity = state
        rng_key1, rng_key2, rng_key3, rng_key4 = jax.random.split(rng_key, 4)

        # Sample a random direction of unit length
        v = metric.sample_unit_ball(rng_key1, x)

        def haussdorff_density(x):
            return logdensity_fn(x) - 0.5 * metric.log_determinant_fn(x)

        def target_vectorized_fn(thetas):
            x_new, _ = metric.geodesic_vectorized_fn(x, v, thetas)
            # Hausdorff density
            logdensity = jax.vmap(haussdorff_density)(x_new)
            return x_new, logdensity

        def target_fn(theta):
            x_new, _ = metric.geodesic_fn(x, v, theta)
            logdensity = haussdorff_density(x_new)
            return x_new, logdensity

        logthreshold = logdensity + jnp.log(jax.random.uniform(rng_key2))
        boundaries, num_reject1 = step_out(
            rng_key3, target_vectorized_fn, logthreshold, w, m
        )

        proposal, (num_reject2, thetas) = shrinkage(
            rng_key4,
            target_fn,
            logthreshold,
            boundaries,
            max_shrinkage=max_shrinkage,
        )

        return proposal, SliceInfo(
            num_reject1, num_reject2, logthreshold, boundaries, thetas
        )

    return kernel


class geodesic_slice_sampler:

    init = staticmethod(init)
    build_kernel = staticmethod(build_shrinkage_kernel)

    def __new__(
        cls,
        logdensity_fn: Callable,
        step_size: float,
        max_stepouts: int,
        metric: Metric,
        max_shrinkage: int = 100,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(
            w=step_size,
            m=max_stepouts,
            metric=metric,
            max_shrinkage=max_shrinkage,
        )

        def init_fn(position: ArrayLikeTree, rng_key=None):
            del rng_key
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(rng_key, state, logdensity_fn)

        return SamplingAlgorithm(init_fn, step_fn)


def step_out(rng_key, target_fn, logthreshold, w, m):
    rng_key1, rng_key2 = jax.random.split(rng_key)

    # Initialization
    u = jax.random.uniform(rng_key1) * w
    l = -u
    r = l + w
    iota = jax.random.choice(rng_key2, jnp.arange(1, m + 1))

    def bound_computation(bound, w, iota):
        steps = jnp.arange(m)
        positions = bound + w * steps
        mask = steps + 2 <= iota
        _, targets = target_fn(positions)
        targets = targets * mask - jnp.inf * (~mask)
        valid = targets > logthreshold
        # when no valid step is found, return -1
        max_idx = jnp.where(jnp.any(valid), m - jnp.argmax(jnp.flip(valid)) - 1, -1)
        return bound + (max_idx + 1) * w, jnp.sum(valid)

    l_final, num_reject_left = bound_computation(l, -w, iota)
    r_final, num_reject_right = bound_computation(r, w, m + 1 - iota)

    # Total number of rejections
    num_reject = num_reject_left + num_reject_right

    return (l_final, r_final), num_reject


def shrinkage(rng_key, target_fn, logthreshold, boundaries, max_shrinkage):

    (l, r) = boundaries
    rng_key1, rng_key = jax.random.split(rng_key)
    theta_h = jax.random.uniform(rng_key1, minval=0, maxval=r - l)
    initial_theta = theta_h + (l - r) * (theta_h > r)
    theta_min = theta_h
    theta_max = theta_h

    num_reject = 0

    initial_proposal, initial_logdensity_proposal = target_fn(initial_theta)
    initial_proposal = initial_proposal.ravel()
    initial_logdensity_proposal = jnp.squeeze(initial_logdensity_proposal)

    def cond_fn(state):
        (
            _,
            _,
            _,
            logdensity_proposal,
            num_reject,
            _,
        ) = state
        return jnp.logical_and(
            logdensity_proposal <= logthreshold, num_reject < max_shrinkage
        )

    def body_fn(state):
        (
            rng_key,
            theta,
            proposal,
            logdensity_proposal,
            num_reject,
            others,
        ) = state

        (theta_min, theta_max, theta_h) = others

        logical_condition = jnp.logical_and(theta_h >= theta_min, theta < r - l)
        theta_min = jnp.where(logical_condition, theta_h, theta_min)
        theta_max = jnp.where(logical_condition, theta_max, theta_h)

        rng_key1, rng_key = jax.random.split(rng_key)
        theta_h = jax.random.uniform(
            rng_key1, minval=0.0, maxval=theta_max + r - l - theta_min
        )
        theta_h = jnp.where(
            theta_h > theta_max, theta_h - theta_max + theta_min, theta_h
        )

        theta = theta_h - (r - l) * (theta_h > r)
        proposal, logdensity_proposal = target_fn(theta)
        proposal = proposal.ravel()
        logdensity_proposal = jnp.squeeze(logdensity_proposal)
        others = (theta_min, theta_max, theta_h)
        return (
            rng_key,
            theta,
            proposal,
            logdensity_proposal,
            num_reject + 1,
            others,
        )

    (_, _, proposal, logdensity_proposal, num_reject, thetas) = lax.while_loop(
        cond_fn,
        body_fn,
        (
            rng_key,
            initial_theta,
            initial_proposal,
            initial_logdensity_proposal,
            num_reject,
            (theta_min, theta_max, theta_h),
        ),
    )
    # Default back to initial position if max_shrinkage is reached
    reset_condition = num_reject == max_shrinkage
    proposal = jnp.where(reset_condition, initial_proposal, proposal)
    logdensity_proposal = jnp.where(
        reset_condition, initial_logdensity_proposal, logdensity_proposal
    )

    return SliceState(proposal, logdensity_proposal), (num_reject, thetas)
