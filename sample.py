import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import blackjax
from utils import (
    evaluate,
    get_reference_draws,
    set_model,
    get_draws,
    get_sampler,
)
from plotting import plot_euclidean, plot_onedim, get_plot, get_contours, plot_chains
from omegaconf import OmegaConf
import hydra

jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))

    manifold_config = cfg.manifold
    sampler_config = cfg.sampler
    repeats = cfg.repeats
    seed = cfg.rng_key
    rng_key = jr.key(seed)
    # manifold conf
    run_evaluation = manifold_config.run_evaluation
    make_plots = manifold_config.make_plots
    num_samples = manifold_config.num_samples
    num_chains = manifold_config.num_chains
    burnin = manifold_config.burnin
    thinning = manifold_config.thinning
    metric_type = manifold_config.metric
    dim = manifold_config.dim
    model_type = manifold_config.model
    sub_name = manifold_config.sub_name
    # sampler conf
    sampler_type = sampler_config.sampler_type

    model, dim = set_model(manifold_config, dim)
    manifold_config.dim = dim

    logdensity_fn = model.logdensity_fn

    sampler = get_sampler(
        model,
        manifold_config,
        sampler_config,
    )

    samples_tensor, info, elapsed_time = get_draws(
        rng_key,
        sampler,
        manifold_config,
        sampler_config,
    )

    # Remove burnin and thinning
    samples_tensor = samples_tensor[burnin::thinning]
    print("Samples shape", samples_tensor.shape)
    samples = samples_tensor.reshape(num_samples, dim)
    true_samples = get_reference_draws(model, model_type, num_samples, sub_name)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if make_plots:
        col_index = -1 if model_type == "funnel" else 1
        if sampler_type not in ["rla", "map_agss"]:
            plot_chains(samples_tensor, output_dir, "chains", dims=[0, col_index])
        if dim == 1:
            xlim = model.xlim
            plot_onedim(
                logdensity_fn,
                true_samples,
                samples,
                xlim,
                output_dir,
                sampler_type,
                num_grids=200,
                show_fig=False,
            )

        if dim >= 2:
            true_samples_reduced = true_samples[:, [0, col_index]]
            samples_reduced = samples[:, [0, col_index]]
            xlim = model.xlim
            ylim = model.ylim
            plot_euclidean(
                model,
                true_samples_reduced,
                samples_reduced,
                xlim,
                ylim,
                dim,
                output_dir,
                name=f"{sampler_type}-{metric_type}",
                show_fig=False,
            )
            if model_type in [
                "gaussian",
                "funnel",
                "squiggle",
                "rosenbrock",
                "generalmixture",
            ]:
                true_dist_levels = model.true_dist_levels
                # 2 dim model for marginal plots
                model2_dim, _ = set_model(manifold_config, 2)
                contours = get_contours(xlim, ylim, model2_dim.logdensity_fn)
                figsize = (6, 6)
                true_dist_colors = ["black", "black", "black"]
                get_plot(
                    model_type,
                    samples_reduced,
                    contours,
                    xlim=xlim,
                    ylim=ylim,
                    hat_theta=None,
                    figsize=figsize,
                    file_name=f"{output_dir}/{sampler_type}_marginal.png",
                    true_dist_levels=true_dist_levels,
                    true_dist_colors=true_dist_colors,
                    use_latex=False,
                )
        if model_type in ["twogaussians"]:
            # View the weights of the two gaussians
            import matplotlib.pyplot as plt

            plt.hist(samples[:, 0], bins=100, alpha=0.5, label="samples")
            plt.hist(true_samples[:, 0], bins=100, alpha=0.5, label="true samples")
            plt.legend()
            plt.savefig(f"{output_dir}/weights.png")

    if run_evaluation:
        repeats = 5
        ess = blackjax.ess(samples_tensor, chain_axis=1, sample_axis=0)
        if sampler_type in ["agss", "map_agss"]:
            avg_stepout = float(info.num_reject_stepout.reshape(-1).mean())
            avg_shrinkage_rejections = float(
                info.num_reject_shrinkage.reshape(-1).mean()
            )
            max_shrinkage_iterations = float(
                jnp.sum(
                    info.num_reject_shrinkage.reshape(-1)
                    == sampler_config.max_shrinkage
                )
            )
        elif sampler_type in ["meta_agss"]:
            avg_stepout = float(info.info_meta.num_reject_stepout.reshape(-1).mean())
            avg_shrinkage_rejections = float(
                info.info_meta.num_reject_shrinkage.reshape(-1).mean()
            )
            max_shrinkage_iterations = float(
                jnp.sum(
                    info.info_meta.num_reject_shrinkage.reshape(-1)
                    == sampler_config.max_shrinkage
                )
            )

        else:
            avg_stepout = None
            avg_shrinkage_rejections = None
            max_shrinkage_iterations = None

        if model_type == "generalmixture":
            # save crossing for each chain
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

            line_crossings = jax.vmap(count_line_crossings, in_axes=1)(samples_tensor)
            line_crossings_avg = line_crossings.sum() / num_samples
            print("Line crossings per 100 iters:", line_crossings_avg * 100)
            line_crossings_avg = float(line_crossings_avg)
        elif model_type == "twogaussians":

            def count_hyperplane_crossings(samples):
                # Compute dot products with the normal vector (1,1,...,1)
                v = jnp.ones(dim)  # Direction vector (1_D - (-1_D))
                projections = jnp.dot(samples, v)  # Project samples onto v
                # Count sign changes in the projection (crossing the hyperplane at origin)
                sign_changes = jnp.sum(jnp.diff(jnp.sign(projections)) != 0)
                return sign_changes

            line_crossings = jax.vmap(count_hyperplane_crossings, in_axes=1)(
                samples_tensor
            )
            line_crossings_avg = line_crossings.sum() / num_samples
            print("Line crossings per 100 iters:", line_crossings_avg * 100)
            line_crossings_avg = float(line_crossings_avg)
        else:
            line_crossings_avg = None

        sampling_stats = {
            "ess": ess.tolist(),
            "elapsed_time": float(elapsed_time),
            "avg_stepout": avg_stepout,
            "avg_shirnkage_rejections": avg_shrinkage_rejections,
            "max_shrinkage_iterations": max_shrinkage_iterations,
            "line_crossings": line_crossings_avg,
        }
        with open(f"{output_dir}/stats.json", "w") as f:
            json.dump(sampling_stats, f, indent=4)

        distances1, distances2 = evaluate(rng_key, samples, true_samples, repeats)
        np.save(f"{output_dir}/distances1.npy", distances1)
        np.save(f"{output_dir}/distances2.npy", distances2)
        if model_type in [
            "funnel",
            "rosenbrock",
            "squiggle",
            "gaussian",
            "twogaussians",
        ]:
            col_index = -1 if model_type == "funnel" else 0
            distances_marginal1, distances_marginal2 = evaluate(
                rng_key,
                samples[:, col_index].reshape(-1, 1),
                true_samples[:, col_index].reshape(-1, 1),
                repeats,
            )
            np.save(f"{output_dir}/distances_marginal1.npy", distances_marginal1)
            np.save(f"{output_dir}/distances_marginal2.npy", distances_marginal2)

            del distances_marginal1, distances_marginal2

        del distances1, distances2

        if model_type in ["generalmixture"]:
            # Save all samples (needed for evaluation)
            np.save(f"{output_dir}/samples.npy", samples_tensor)


if __name__ == "__main__":
    my_app()
