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
    count_line_crossings,
    count_hyperplane_crossings,
    compute_proportions,
    stein_disc,
    count_middle_sample_crossings,
    compute_middle_sample_proportions,
)
from plotting import (
    plot_euclidean,
    plot_onedim,
    get_plot,
    get_contours,
    plot_chains,
    plot_weights,
    plot_field,
)
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
    if model_type not in ["phifour"]:
        true_samples = get_reference_draws(model, model_type, num_samples, sub_name)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if make_plots:
        col_index = -1 if model_type == "funnel" else 1
        if sampler_type not in ["rla"]:
            plot_chains(samples_tensor, output_dir, "chains", dims=[0, col_index])
        if model_type not in ["phifour"]:
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
            # One dim histogram
            plot_weights(samples, true_samples, output_dir)
        if model_type in ["phifour"]:
            # Custom plot the Field
            plot_field(samples, dim, output_dir)

    if run_evaluation:
        repeats = 5
        ess = blackjax.ess(samples_tensor, chain_axis=1, sample_axis=0)
        if sampler_type in ["magss"]:
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
        elif sampler_type in ["meta_magss"]:
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
            line_crossings = jax.vmap(count_line_crossings, in_axes=1)(samples_tensor)
            line_crossings_avg = line_crossings.sum() / num_samples
            print("Line crossings per 100 iters:", line_crossings_avg * 100)
            line_crossings_avg = float(line_crossings_avg)
            proportions = None
            # Save all samples
            np.save(f"{output_dir}/samples.npy", samples_tensor)
        elif model_type in ["twogaussians", "twogaussiansequal"]:
            line_crossings = jax.vmap(count_hyperplane_crossings, in_axes=1)(
                samples_tensor
            )
            line_crossings_avg = line_crossings.sum() / num_samples
            proportions = compute_proportions(samples)
            print("Line crossings per 100 iters:", line_crossings_avg * 100)
            print("Proportions:", proportions)
            line_crossings_avg = float(line_crossings_avg)
        elif model_type == "phifour":
            line_crossings = jax.vmap(count_middle_sample_crossings, in_axes=1)(
                samples_tensor
            )
            line_crossings_avg = line_crossings.sum() / num_samples
            line_crossings_avg = float(line_crossings_avg)
            proportions = compute_middle_sample_proportions(samples)
            print("Line crossings per 100 iters:", line_crossings_avg * 100)
            print("Proportions:", proportions)

        else:
            line_crossings_avg = None
            proportions = None

        if model_type not in ["phifour"]:
            stein_u, stein_v = None, None
            distances1, distances2 = evaluate(rng_key, samples, true_samples, repeats)
            wass_avg = float(np.mean(distances1))
            wass_std = float(np.std(distances1))
            # Save full numpy arrays
            np.save(f"{output_dir}/distances1.npy", distances1)
            np.save(f"{output_dir}/distances2.npy", distances2)

        else:
            wass_avg = None
            wass_std = None
            stein_u, stein_v = stein_disc(samples, logdensity_fn)
            stein_u = float(stein_u)
            stein_v = float(stein_v)

        sampling_stats = {
            "ess": ess.tolist(),
            "elapsed_time": float(elapsed_time),
            "avg_stepout": avg_stepout,
            "avg_shirnkage_rejections": avg_shrinkage_rejections,
            "max_shrinkage_iterations": max_shrinkage_iterations,
            "line_crossings": line_crossings_avg,
            "proportions": proportions,
            "wasserstein_distance": [wass_avg, wass_std],
            "stein": [stein_u, stein_v],
        }
        with open(f"{output_dir}/stats.json", "w") as f:
            json.dump(sampling_stats, f, indent=4)

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


if __name__ == "__main__":
    my_app()
