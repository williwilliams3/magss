import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.base import SamplingAlgorithm
from typing import Callable, NamedTuple, Any
from blackjax.types import ArrayLikeTree, PRNGKey, ArrayTree


class DIGSState(NamedTuple):
    position: ArrayTree


class DIGSInfo(NamedTuple):
    sampler_info: NamedTuple
    acceptance_rate: float


def init(position):
    return DIGSState(position)


def build_kernel(get_proposal: bool):

    if get_proposal:

        def get_body_fn(logdensity_fn, sampler_fn, alg_steps):
            def body_fn(state, rng_key):
                (x, alpha) = state
                sigma = jnp.sqrt(1 - jnp.square(alpha))
                rng_key, rng_key1, rng_key2, rng_key3, rng_key4, rng_key5 = jr.split(
                    rng_key, 6
                )
                x_tilde = alpha * x + sigma * jr.normal(rng_key1, x.shape)
                # Proposal step
                x_init_prime = x_tilde / alpha + (sigma / alpha) * jr.normal(
                    rng_key2, x.shape
                )
                new_logdensity_fn = (
                    lambda x: logdensity_fn(x)
                    - jnp.sum(jnp.square(x_tilde - alpha * x)) / 2.0 / sigma**2
                )
                proposal_logpdf_fn = (
                    lambda x: -jnp.sum(jnp.square(x - x_tilde / alpha))
                    / 2.0
                    / (sigma / alpha) ** 2
                )
                log_a_init = (
                    new_logdensity_fn(x_init_prime)
                    - new_logdensity_fn(x)
                    + proposal_logpdf_fn(x)
                    - proposal_logpdf_fn(x_init_prime)
                )

                # MH step
                u = jr.uniform(rng_key3)
                x_init = jnp.where(jnp.log(u) < log_a_init, x_init_prime, x)

                # Score based MCMC
                sampling_alg = sampler_fn(new_logdensity_fn)

                state_sba = sampling_alg.init(position=x_init, rng_key=rng_key4)
                keys_sba = jr.split(rng_key5, alg_steps)

                def one_step_sba(state, rng_key):
                    state, info = sampling_alg.step(rng_key, state)
                    return state, (state, info)

                state_sba, (state_sbas, info_sba) = jax.lax.scan(
                    one_step_sba, state_sba, keys_sba
                )

                return (state_sba.position, alpha), (
                    state_sbas.position,
                    DIGSInfo(info_sba, jnp.minimum(jnp.exp(log_a_init), 1.0)),
                )

            return body_fn

    else:

        def get_body_fn(logdensity_fn, sampler_fn, alg_steps):
            def body_fn(state, rng_key):
                (x, alpha) = state
                sigma = jnp.sqrt(1 - jnp.square(alpha))  # VE Scheme
                rng_key, rng_key1, rng_key2, rng_key3 = jr.split(rng_key, 4)
                x_tilde = alpha * x + sigma * jr.normal(rng_key1, x.shape)
                new_logdensity_fn = (
                    lambda x: logdensity_fn(x)
                    - jnp.sum(jnp.square(x_tilde - alpha * x)) / 2.0 / sigma**2
                )

                # Score based MCMC
                x_init = x
                sampling_alg = sampler_fn(new_logdensity_fn)
                state_sba = sampling_alg.init(position=x_init, rng_key=rng_key2)
                keys_sba = jr.split(rng_key3, alg_steps)

                def one_step_sba(state, rng_key):
                    state, info = sampling_alg.step(rng_key, state)
                    return state, (state, info)

                state_sba, (state_sbas, info_sba) = jax.lax.scan(
                    one_step_sba, state_sba, keys_sba
                )

                return (state_sba.position, alpha), (
                    state_sbas.position,
                    DIGSInfo(info_sba, 1.0),
                )

            return body_fn

    def kernel(
        rng_key,
        digs_state,
        logdensity_fn,
        sampler_fn,
        num_alphas,
        alphas,
        gibbs_sweeps,
        alg_steps,
    ):
        body_fn = get_body_fn(logdensity_fn, sampler_fn, alg_steps)

        def wrapper_body_fn(state, rng_key):
            (x, i) = state
            alpha = alphas[i]
            (x, _), (xs, info) = jax.lax.scan(
                body_fn, (x, alpha), jr.split(rng_key, gibbs_sweeps)
            )
            return (x, i + 1), (xs, info)

        x = digs_state.position
        state, (xs, info) = jax.lax.scan(
            wrapper_body_fn, (x, 0), jr.split(rng_key, num_alphas)
        )
        (x, _) = state

        return DIGSState(x), (xs, info)

    return kernel


class digs:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(
        cls,
        logdensity_fn: Callable,
        sampler_fn,
        num_alphas,
        alphas,
        gibbs_sweeps,
        alg_steps,
        get_proposal: bool,
    ) -> SamplingAlgorithm:

        kernel = cls.build_kernel(get_proposal)

        def init_fn(position: ArrayLikeTree, rng_key):
            del rng_key
            return cls.init(position)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                sampler_fn,
                num_alphas,
                alphas,
                gibbs_sweeps,
                alg_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)
