import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.base import SamplingAlgorithm
from typing import Callable
from blackjax.types import ArrayLikeTree, PRNGKey


# In principle the swapping operations do not involve extra computations


def get_tempered_logdensity_fn(logdensity_fn, inv_temperature):
    return lambda x: logdensity_fn(x) * inv_temperature


def individual_step(sampler_fn, logdensity_fn, inv_temperature, key, position):
    sampler = sampler_fn(get_tempered_logdensity_fn(logdensity_fn, inv_temperature))
    # Could in principle re-use the previous
    state = sampler.init(position)
    new_state, info = sampler.step(key, state)
    return new_state.position, info


def swap_elements(arr, i):
    perm = jnp.arange(arr.shape[0])
    perm = perm.at[i].set(i + 1)
    perm = perm.at[i + 1].set(i)
    return arr[perm]


def maybe_swap(i, position, inv_temperatures, logdensity_fn, rng_key):
    # Could in principle re-use the previous
    position1 = position[i]
    position2 = position[i + 1]
    inv_temperature1 = inv_temperatures[i]
    inv_temperature2 = inv_temperatures[i + 1]
    log_a = (inv_temperature1 - inv_temperature2) * (
        logdensity_fn(position2) - logdensity_fn(position1)
    )

    # MH step
    u = jr.uniform(rng_key)

    return jax.lax.cond(
        jnp.log(u) < log_a, lambda: swap_elements(position, i), lambda: position
    )


def init(position, num_temperatures):
    return jnp.tile(position, (num_temperatures,) + (1,) * position.ndim)


def build_kernel():

    def kernel(
        rng_key: PRNGKey,
        state,
        sampler_fn,
        logdensity_fn: Callable,
        inv_temperatures,
        num_temperatures,
        alg_steps,
    ):

        def body_fn(state, x):
            (rng_key, i) = x
            state = maybe_swap(i, state, inv_temperatures, logdensity_fn, rng_key)
            return state, None

        def wrapper_body_fn(state, x):
            (rng_key, i) = x
            # One extra one for swapping; do not need it for the last iteration but still leave it here
            keys = jr.split(rng_key, num_temperatures + 1)

            state, info = jax.vmap(
                lambda inv_temperature, key, position: individual_step(
                    sampler_fn, logdensity_fn, inv_temperature, key, position
                ),
                in_axes=(0, 0, 0),
            )(inv_temperatures, keys[:-1], state)

            def false_fn(key, state):
                keys = jr.split(key, num_temperatures - 1)
                # No info for swapping for now
                state, _ = jax.lax.scan(
                    body_fn, state, (keys, jnp.arange(num_temperatures - 1))
                )
                return state

            state = jax.lax.cond(
                i == alg_steps - 1,
                lambda _, state: state,
                false_fn,
                keys[-1],
                state,
            )

            return state, (state, info)

        state, (states, info) = jax.lax.scan(
            wrapper_body_fn,
            state,
            (jr.split(rng_key, alg_steps), jnp.arange(alg_steps)),
        )

        return state, (states, info)

    return kernel


class pt:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(
        cls,
        logdensity_fn: Callable,
        sampler_fn,
        num_temperatures,
        inv_temperatures,
        alg_steps,
    ) -> SamplingAlgorithm:

        kernel = cls.build_kernel()

        def init_fn(position: ArrayLikeTree, rng_key):
            del rng_key
            return cls.init(position, num_temperatures)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                sampler_fn,
                logdensity_fn,
                inv_temperatures,
                num_temperatures,
                alg_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)
