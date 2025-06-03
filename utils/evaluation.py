import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import ot

from typing import Tuple


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


def count_line_crossings(samples: jnp.ndarray):
    x_points = jnp.array([-2, -1.8])
    y_points = jnp.array([0, 1])
    m = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
    b = y_points[0] - m * x_points[0]

    line_y = m * samples[:, 0] + b
    below_line = samples[:, 1] < line_y
    above_line = samples[:, 1] >= line_y
    transitions_up = jnp.sum(below_line[:-1] & above_line[1:])
    transitions_down = jnp.sum(above_line[:-1] & below_line[1:])
    return transitions_up + transitions_down


def count_hyperplane_crossings(samples):
    dim = samples.shape[1]
    # Compute dot products with the normal vector (1,1,...,1)
    v = jnp.ones(dim)  # Direction vector (1_D - (-1_D))
    projections = jnp.dot(samples, v)  # Project samples onto v
    # Count sign changes in the projection (crossing the hyperplane at origin)
    sign_changes = jnp.sum(jnp.diff(jnp.sign(projections)) != 0)
    return sign_changes


def count_middle_sample_crossings(samples):
    """
    Computes the number of sign changes along the middle dimension of `samples`

    Args:
        samples (jnp.ndarray): Input array of shape (N, D) where N is the number of samples and D is the dimension.

    Returns:
        int: number of sign changes
    """
    dim = samples.shape[1]
    mid_dim = dim // 2  # Floor(D/2)

    # Extract the middle dimension
    middle_values = samples[:, mid_dim]

    # Count sign changes
    sign_changes = jnp.sum(jnp.diff(jnp.sign(middle_values)) != 0)

    # Compute proportions
    num_positive = jnp.sum(middle_values > 0)
    total = middle_values.shape[0]
    prop_positive = num_positive / total

    return sign_changes


def compute_middle_sample_proportions(samples):
    """
    Computes the proportions of samples above and below the middle dimension
    Args:
        samples (jnp.ndarray): Input array of shape (N, D) where N is the number of samples and D is the dimension.
    Returns:
        list: proportions of samples above and below the middle dimension
    """
    dim = samples.shape[1]
    mid_dim = dim // 2  # Floor(D/2)

    # Extract the middle dimension
    middle_values = samples[:, mid_dim]

    # Compute proportions
    num_positive = jnp.sum(middle_values > 0)
    total = middle_values.shape[0]
    prop_positive = num_positive / total

    return [float(prop_positive), float(1 - prop_positive)]


def compute_proportions(samples):
    dim = samples.shape[1]
    center_1 = -jnp.ones(dim)
    center_2 = jnp.ones(dim)

    # Compute distances to the two centers
    d1 = jnp.linalg.norm(samples - center_1, axis=1)
    d2 = jnp.linalg.norm(samples - center_2, axis=1)

    # Count samples closer to each center
    closer_to_1 = jnp.sum(d1 < d2)
    closer_to_2 = jnp.sum(d2 <= d1)  # Tie-breaking favors w2

    # Compute proportions
    num_samples = samples.shape[0]
    w1 = float(closer_to_1 / num_samples)
    w2 = float(closer_to_2 / num_samples)

    return [w1, w2]


def stein_disc(X, logprob_fn, beta=-1 / 2) -> Tuple:
    """Stein discrepancy with inverse multi-quadric kernel,
    i.e. (1 + (x - x')T(x - x')) ** beta
    returns U-Statistic (unbiased) and V-statistic (biased)
    """

    T, d = X.shape
    sub = lambda x, x_: x - x_
    grad = jax.grad(logprob_fn)
    beta = -beta

    def disc(x, x_):
        diff = sub(x, x_)
        dot_prod = jnp.dot(diff, diff)
        dx = grad(x)
        dx_ = grad(x_)
        return (
            -4 * beta * (beta + 1) * dot_prod / ((1 + dot_prod) ** (beta + 2))
            + 2 * beta * (d + jnp.dot(dx - dx_, diff)) / ((1 + dot_prod) ** (1 + beta))
            + jnp.dot(dx, dx_) / ((1 + dot_prod) ** beta)
        )

    _disc = jax.vmap(disc, (None, 0))
    mc_sum = jax.lax.map(lambda x: _disc(x, X).sum(), X).sum()
    return (mc_sum - jax.vmap(lambda x: disc(x, x))(X).sum()) / (
        T * (T - 1)
    ), mc_sum / T**2
