import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple, Callable, Tuple
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from blackjax.base import SamplingAlgorithm


class SliceState(NamedTuple):
    position: ArrayTree
    logdensity: float


class SliceInfo(NamedTuple):
    num_reject_stepout: int
    num_reject_shrinkage: int
    logthreshold: float
    boundaries: ArrayLikeTree
    thetas: ArrayLikeTree


def init(position: ArrayLikeTree, logdensity_fn: Callable):
    logdensity = logdensity_fn(position)
    return SliceState(position, logdensity)


def build_shrinkage_kernel(
    w: float, m: int, do_shrinkage: bool = True, max_shrinkage: int = 100
):

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logdensity_fn: Callable,
    ):
        x, logdensity = state
        rng_key1, rng_key2, rng_key3 = jax.random.split(rng_key, 3)

        target_fn = lambda theta: (x + theta, logdensity_fn(x + theta))

        logthreshold = logdensity + jnp.log(jax.random.uniform(rng_key1))
        boundaries, num_reject1 = step_out(rng_key2, target_fn, logthreshold, w, m)
        if do_shrinkage:
            proposal, (num_reject2, thetas) = shrinkage(
                rng_key3, target_fn, logthreshold, boundaries, max_shrinkage
            )
        else:
            proposal, (num_reject2, thetas) = no_shrinkage(
                rng_key3, target_fn, logthreshold, boundaries
            )

        return proposal, SliceInfo(
            num_reject1, num_reject2, logthreshold, boundaries, thetas
        )

    return kernel


class univariate_slice_sampler:

    init = staticmethod(init)
    build_kernel = staticmethod(build_shrinkage_kernel)

    def __new__(
        cls,
        logdensity_fn: Callable,
        step_size: float,
        max_stepouts: int,
        do_shrinkage: bool = True,
        max_shrinkage: int = jnp.inf,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(
            w=step_size,
            m=max_stepouts,
            do_shrinkage=do_shrinkage,
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
        targets = jax.vmap(lambda theta: target_fn(theta)[1])(
            positions
        ) * mask - jnp.inf * (~mask)
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

        (theta_min, theta_max) = lax.cond(
            jnp.logical_and(theta_h >= theta_min, theta < r - l),
            lambda: (theta_h, theta_max),
            lambda: (theta_min, theta_h),
        )

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

    (_, theta, proposal, logdensity_proposal, num_reject, thetas) = lax.while_loop(
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
    proposal = jnp.where(
        num_reject == max_shrinkage,
        initial_proposal,
        proposal,
    )
    logdensity_proposal = jnp.where(
        num_reject == max_shrinkage,
        initial_logdensity_proposal,
        logdensity_proposal,
    )

    return SliceState(proposal, logdensity_proposal), (num_reject, thetas)


def no_shrinkage(
    rng_key, target_fn: Callable, logthreshold: float, boundaries: Tuple[float, float]
):
    """
    Sampling loop without shrinkage: Samples uniformly within boundaries (l, r) until acceptance.

    Parameters:
    - rng_key: PRNGKey for reproducibility.
    - target_fn: Callable that computes the log-density and returns (proposal, logdensity_proposal).
    - logthreshold: Logarithm of the slice threshold.
    - boundaries: Tuple of (l, r), the interval to sample from.

    Returns:
    - SliceState: A NamedTuple containing the accepted proposal and its log-density.
    - int: The number of rejected samples before acceptance.
    """
    l, r = boundaries  # Extract boundaries
    rng_key1, rng_key = jax.random.split(rng_key)
    theta = jax.random.uniform(rng_key1, minval=l, maxval=r)
    proposal, logdensity_proposal = target_fn(theta)
    proposal = proposal.ravel()
    logdensity_proposal = jnp.squeeze(logdensity_proposal)
    state = (
        rng_key,  # RNG key
        0,  # Number of rejections
        logdensity_proposal,  # Current log-density
        (proposal, theta),  # Placeholder for proposal and theta
    )

    def cond_fn(state):
        """Continue until the sampled point is accepted."""
        _, _, logdensity_proposal, _ = state
        return logdensity_proposal <= logthreshold

    def body_fn(state):
        """Sample a new candidate and evaluate its log-density."""
        rng_key, num_reject, logdensity_proposal, _ = state

        # Sample uniformly within the interval [l, r]
        rng_key1, rng_key = jax.random.split(rng_key)
        theta = jax.random.uniform(rng_key1, minval=l, maxval=r)

        # Evaluate log-density of the proposal
        proposal, logdensity_proposal = target_fn(theta)
        proposal = proposal.ravel()
        logdensity_proposal = jnp.squeeze(logdensity_proposal)

        return rng_key, num_reject + 1, logdensity_proposal, (proposal, theta)

    # Run the while loop
    _, num_reject, logdensity_proposal, (proposal, theta) = lax.while_loop(
        cond_fn, body_fn, state
    )

    # Return the accepted state and number of rejections
    return SliceState(proposal, logdensity_proposal), (num_reject, theta)
