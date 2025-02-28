import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import pytest
import time
from mcmc.univariate_slice_sampler import step_out as step_out_new

# Note:  pytest -s  geoslice/test/test_stepout.py     shows the print statements
#        pytest geoslice/test/test_stepout.py          does not show the print statements


@pytest.fixture
def setup():
    # Define a sample target function
    logdensity_fn = lambda x: -0.5 * x**2
    x = 1.0
    target_fn = lambda theta: (x + theta, logdensity_fn(x + theta))
    threshold = -1.0
    rng_key = jax.random.key(42)
    return rng_key, target_fn, threshold


def test_step_out_equivalence(setup):
    rng_key, target_fn, threshold = setup

    # Set global parameters for the test
    w = 0.1
    m = 10

    # Call both functions
    start_time_old = time.time()
    old_output = step_out_old(rng_key, target_fn, threshold, w, m)
    end_time_old = time.time()

    start_time_new = time.time()
    new_output = step_out_new(rng_key, target_fn, threshold, w, m)
    end_time_new = time.time()

    # Compare running times

    print(f"Old Runtime: {end_time_old - start_time_old:.6f} seconds")
    print(f"New Runtime: {end_time_new - start_time_new:.6f} seconds")
    print(f"Num. rejections: {old_output[1]}")

    # Compare outputs
    assert jnp.allclose(
        old_output[0][0], new_output[0][0]
    ), f"Left bounds do not match {old_output[0][0]}!={new_output[0][0]}"
    assert jnp.allclose(
        old_output[0][1], new_output[0][1]
    ), f"Right bounds do not match {old_output[0][1]}!={new_output[0][1]}"
    assert (
        old_output[1] == new_output[1]
    ), f"Number of rejections do not match {old_output[1]}!={new_output[1]}"


def step_out_old(rng_key, target_fn, threshold, w, m):

    rng_key1, rng_key2 = jax.random.split(rng_key)
    u = jax.random.uniform(rng_key1) * w
    l = -u
    r = l + w
    iota = jax.random.choice(rng_key2, jnp.arange(1, m + 1))
    i = 2
    j = 2

    num_reject = 0

    def cond_fn(state):
        (l, i, _) = state
        return jnp.logical_and(i <= iota, target_fn(l)[1] > threshold)

    def body_fn(state):
        (l, i, num_reject) = state
        return (l - w, i + 1, num_reject + 1)

    (l, _, num_reject) = jax.lax.while_loop(cond_fn, body_fn, (l, i, num_reject))

    def cond_fn(state):
        (r, j, _) = state
        return jnp.logical_and(j <= m + 1 - iota, target_fn(r)[1] > threshold)

    def body_fn(state):
        (r, j, num_reject) = state
        return (r + w, j + 1, num_reject + 1)

    (r, _, num_reject) = jax.lax.while_loop(cond_fn, body_fn, (r, j, num_reject))

    return (l, r), num_reject


def step_out_new(rng_key, target_fn, logthreshold, w, m):
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
