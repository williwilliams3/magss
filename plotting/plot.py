import matplotlib.pyplot as plt
import jax.random as jr
import jax.numpy as jnp
import jax


def plot_euclidean(
    model,
    exact_samples,
    samples,
    xlim,
    ylim,
    dim,
    output_dir,
    name,
    num_grids=200,
    show_fig=False,
):
    if dim == 2:
        xs = jnp.linspace(model.xlim[0], model.xlim[1], num_grids)
        ys = jnp.linspace(model.ylim[0], model.ylim[1], num_grids)
        Xs, Ys = jnp.meshgrid(xs, ys)
        positions = jnp.stack([Xs, Ys], axis=2)
        Zs = jax.vmap(jax.vmap(model.logdensity_fn, in_axes=0), in_axes=0)(positions)

    # Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(exact_samples[:, 0], exact_samples[:, 1], alpha=0.01)
    if dim == 2:
        axs[0].contour(Xs, Ys, Zs, levels=30)
    axs[0].set_title("True")
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[1].scatter(samples[:, 0], samples[:, 1], alpha=0.01)
    if dim == 2:
        axs[1].contour(Xs, Ys, Zs, levels=30)
    axs[1].set_title("MCMC")
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    if show_fig:
        plt.show()
    else:
        plt.savefig(f"{output_dir}/{name}.png", dpi=200)
        plt.close()


def plot_onedim(
    logdensity_fn,
    exact_samples,
    samples,
    xlim,
    output_dir,
    name,
    num_grids=200,
    show_fig=False,
):

    xs = jnp.linspace(xlim[0], xlim[1], num_grids)
    ys = jax.vmap(logdensity_fn)(xs)
    ys = jnp.exp(ys)
    # Plot side by side
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(exact_samples, alpha=0.5, density=True, bins=40)
    ax.plot(xs, ys)
    ax.set_title("True")
    ax.set_xlim(xlim)
    if show_fig:
        plt.show()
    else:
        plt.savefig(f"{output_dir}/{name}_true.png", dpi=200)
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(samples, alpha=0.5, density=True, bins=40)
    ax.plot(xs, ys)
    ax.set_title(name)
    ax.set_xlim(xlim)

    if show_fig:
        plt.show()
    else:
        plt.savefig(f"{output_dir}/{name}_mcmc.png", dpi=200)
        plt.close()


def plot_chains(samples, output_dir, name, dims):
    num_samples, num_chains, dim = samples.shape
    x = jnp.arange(num_samples)

    # Plot chains on dims specified or the single dimension
    for i in range(min(len(dims), dim)):
        if dim == 1:
            d = i
        else:
            d = dims[i]
        plt.figure(figsize=(10, 6))
        for c in range(num_chains):
            plt.plot(x, samples[:, c, d], label=f"Chain {c+1}", alpha=0.6)

        plt.xlabel("Iteration")
        plt.ylabel(f"Chain values (Dimension {d+1})")
        plt.title(f"Chains for Dimension {d+1}")
        plt.legend()
        plt.savefig(f"{output_dir}/{name}_{d+1}.png", dpi=200)
        plt.close()


def plot_weights(samples, true_samples, output_dir):
    plt.figure()
    plt.hist(samples[:, 0], bins=100, alpha=0.5, label="samples")
    plt.hist(true_samples[:, 0], bins=100, alpha=0.5, label="true samples")
    plt.legend()
    plt.savefig(f"{output_dir}/weights.png")


def sub_sample(rng_key, samples, max_samples=1000):
    n_rows, _ = samples.shape
    if n_rows <= max_samples:
        return samples
    if n_rows > max_samples:
        id_samples = jr.choice(rng_key, n_rows, shape=(max_samples,), replace=False)
        return samples[id_samples]


def plot_field(samples, dim, output_dir):
    key = jr.key(42)
    samples = sub_sample(key, samples, max_samples=1000)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=True, sharey=True)
    ax.set_title(r"$\hat{\phi}$")
    ax.set_xlabel(r"$d$")
    ax.set_ylabel(r"$\phi$")
    samples = jnp.pad(samples, ((0, 0), (1, 1)))  # for the phi-four example
    for i in range(samples.shape[0]):
        ax.plot(samples[i], color="red", alpha=0.1)
    plt.setp(ax, xlim=[0, dim + 1], ylim=[-1.6, 1.6])
    plt.savefig(f"{output_dir}/field.png")
