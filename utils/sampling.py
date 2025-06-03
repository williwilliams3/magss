import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import os
import time


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(current_file_path))


def inference_loop(rng_key, num_samples, num_chains, sampler, initial_states):

    @jax.jit
    def _one_step(states, rng_key):

        keys = jr.split(rng_key, num_chains)
        states, info = jax.vmap(sampler.step)(keys, states)
        return states, (states, info)

    keys = jr.split(rng_key, num_samples)
    start_time = time.time()
    _, (states, info) = jax.lax.scan(_one_step, initial_states, keys)
    jax.block_until_ready(states)  # Ensure all computations finish
    end_time = time.time()
    elapsed_time = end_time - start_time

    return states, info, elapsed_time


def get_draws(
    rng_key,
    sampler,
    manifold_config,
    sampler_config,
):
    sampler_type = sampler_config.sampler_type
    model_type = manifold_config.model
    burnin = manifold_config.burnin
    thinning = manifold_config.thinning
    num_samples = manifold_config.num_samples
    num_chains = manifold_config.num_chains
    dim = manifold_config.dim
    rng_key1, rng_key2 = jr.split(rng_key)

    total_num_steps = burnin + (num_samples // num_chains) * thinning
    # Set initial positions
    if model_type == "twogaussians" or model_type == "twogaussiansequal":
        initial_positions = jnp.ones((num_chains, dim)) * (-1.0)
    elif model_type == "phifour":
        # keys = jax.random.split(rng_key, num_chains)
        # initial_positions = jax.vmap(lambda k: jax.random.uniform(k, (dim,)) * 2 - 1)(
        #     keys
        # )
        # initial_positions = jnp.ones((num_chains, dim))
        initial_positions = jnp.ones((num_chains, dim)) * (-1.0)

    else:
        initial_positions = jnp.zeros((num_chains, dim))

    if sampler_type not in ["rla"]:
        init_keys = jr.split(rng_key1, num_chains)
        initial_states = jax.vmap(sampler.init)(
            initial_positions,
            init_keys,
        )
        states, info, elapsed_time = inference_loop(
            rng_key2, total_num_steps, num_chains, sampler, initial_states
        )
        print("Sampling time: ", elapsed_time)

    elif sampler_type in ["rla"]:
        key1, key2 = jr.split(rng_key, 2)
        keys1 = jr.split(key1, total_num_steps * num_chains)
        keys2 = jr.split(key2, total_num_steps * num_chains)
        if model_type in ["gaussian", "squiggle"]:
            x_map = jnp.zeros(dim)
        elif model_type in ["rosenbrock"]:
            x_map = jnp.ones(dim)
        else:
            raise NotImplementedError(
                f"Model {model_type} not implemented for {sampler_type}"
            )
        initial_states = jax.vmap(sampler.init, in_axes=(None, 0))(x_map, keys1)
        start_time = time.time()
        states, info = jax.vmap(sampler.step)(keys2, initial_states)
        jax.block_until_ready(states)  # Ensure all computations finish
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Sampling time: ", elapsed_time)

    # Print information
    if sampler_type.startswith("digs"):
        samples_tensor = states.position
        info = info[1]
        print("Acceptance rate sba", info[0].acceptance_rate.mean())
        print("Acceptance rate mh", info.acceptance_rate.mean())
    elif sampler_type.startswith("nopropdigs"):
        samples_tensor = states.position
        info = info[1]
        print("Acceptance rate sba", info[0].acceptance_rate.mean())
        print("Acceptance rate mh", info.acceptance_rate.mean())
    elif sampler_type.startswith("coupled"):
        samples_tensor = states.position
        info = info[1]
        print("Acceptance rate sba", info[0].acceptance_rate.mean())
        print("Acceptance rate mh", info.acceptance_rate.mean())
    elif sampler_type.startswith("pt"):
        samples_tensor = states[:, :, 0, :]
        info = info[1]
        print("Acceptance rate", info.acceptance_rate[:, :, 0, 0].mean())
    else:
        samples_tensor = states.position
        if sampler_type not in [
            "magss",
            "rla",
            "meta_magss",
        ]:
            print("Acceptance rate", info.acceptance_rate.mean())
        elif sampler_type in ["magss"]:
            print(
                "Max shrinkage iterations",
                jnp.sum(
                    info.num_reject_shrinkage.reshape(-1)
                    == sampler_config.max_shrinkage
                ),
            )
        elif sampler_type == "meta_magss":
            print("Acceptance rate", info.info_sba.acceptance_rate.mean())
            print(
                "Max shrinkage iterations",
                jnp.sum(
                    info.info_meta.num_reject_shrinkage.reshape(-1)
                    == sampler_config.max_shrinkage
                ),
            )
    return samples_tensor, info, elapsed_time


def get_reference_draws(model, name_model, num_samples, sub_name=""):
    if name_model in [
        "funnel",
        "squiggle",
        "rosenbrock",
        "gaussian",
        "twogaussians",
        "twogaussiansequal",
        "ninegaussians",
        "gmm",
        "generalmixture",
    ]:
        rng_key_true = jr.key(42)
        samples_true = np.array(model.sample(rng_key_true, num_samples))
    elif name_model == "banana":
        in_path = f"data/reference_samples/{name_model}/reference_samples.npy"
        in_path = os.path.join(current_directory, in_path)
        samples_true = np.load(in_path)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples]
    elif name_model == "logreg":
        in_path = (
            f"data/reference_samples/{name_model}/{sub_name}/reference_samples.npy"
        )
        in_path = os.path.join(current_directory, in_path)
        samples_true = np.load(in_path)
        n, d = samples_true.shape
        assert num_samples <= n
        samples_true = samples_true[:num_samples]
    return samples_true
