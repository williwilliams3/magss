import jax.numpy as jnp
import jax.random as jr
import numpy as np
import ot


def wasserstein_distance(samples1, samples2, distance_fn):
    M = np.asarray(distance_fn(jnp.asarray(samples1), jnp.asarray(samples2)))
    return ot.emd2([], [], M, numItermax=1e10)


def evaluate(
    rng_key,
    samples,
    true_samples,
    repeats,
    subsample_size=2000,
):
    assert samples.shape == true_samples.shape
    num_samples = samples.shape[0]
    rng_keys = jr.split(rng_key, repeats)
    distances1 = []
    distances2 = []

    for rng_key in rng_keys:

        rng_key1, rng_key2 = jr.split(rng_key)
        indexes1 = jr.choice(rng_key1, jnp.arange(num_samples), (subsample_size,))
        indexes2 = jr.choice(rng_key2, jnp.arange(num_samples), (subsample_size,))
        distance_fn = lambda samples1, samples2: ot.dist(
            samples1, samples2, metric="euclidean"
        )
        distances1.append(
            wasserstein_distance(
                np.asarray(samples[indexes1]),
                np.asarray(true_samples[indexes2]),
                distance_fn,
            )
        )
        distances2.append(
            wasserstein_distance(
                np.asarray(true_samples[indexes1]),
                np.asarray(true_samples[indexes2]),
                distance_fn,
            )
        )

    distances1 = np.array(distances1)
    distances2 = np.array(distances2)
    print(
        f"Wasserstein distance to true samples: {[np.round(np.mean(distances1), 2), np.round(np.std(distances1), 2)]}"
    )
    print(
        f"Wasserstein distance between true samples: {[np.round(np.mean(distances2), 2), np.round(np.std(distances2), 2)]}"
    )

    return distances1, distances2
