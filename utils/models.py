import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import jax.random as jr

from metrics.models import (
    GMM,
    Gaussian,
    TwoGaussians,
    Funnel,
    Squiggle,
    Rosenbrock,
    BayesianLogisticRegression,
    NineGaussians,
    PhiFour,
    GeneralMixture,
)


def set_model(manifold_config, dim):
    model_rng_key = manifold_config.model_rng_key
    model_type = manifold_config.model
    sub_name = manifold_config.sub_name
    if model_type == "gmm":
        model = GMM(
            rng_key=jr.key(model_rng_key),
            dim=dim,
            means=None,
            log_sigma=1.0,
            num_mixtures=40,
        )

    elif model_type == "gaussian":
        model = Gaussian(dim=dim)
    elif model_type == "twogaussians":
        model = TwoGaussians(dim=dim)
    elif model_type == "twogaussiansequal":
        model = TwoGaussians(dim=dim, weights=[0.5, 0.5])
    elif model_type == "ninegaussians":
        model = NineGaussians(dim=dim)
    elif model_type == "funnel":
        model = Funnel(dim=dim)
    elif model_type == "squiggle":
        model = Squiggle(dim=dim)
    elif model_type == "rosenbrock":
        model = Rosenbrock(dim=dim)
    elif model_type == "logreg":
        model = BayesianLogisticRegression(dataset_name=sub_name)
        dim = model.D
    elif model_type == "generalmixture":
        model = GeneralMixture(dim)
    elif model_type == "phifour":
        model = PhiFour(dim)
    else:
        raise ValueError("Invalid model name")
    return model, dim
