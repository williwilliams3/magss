# Metric-Agnostic Geodesic Slice Sampler

## Reproducing Figures

To replicate **Figure 1**, run:
```bash
python magss/plotting/slice_explanation.py
```

To replicate **Figure 2**, run:
```bash
python magss/plotting/hit_and_run.py
```

To replicate **Figure 3**, run:
```bash
python magss/plotting/geodesic_balls.py
```


## Sampling
To sample the **Toy distributions** run for model in funnel, rosenbrock, squiggle:
```bash
python magss/sample.py  \
sampler=agss \
manifold.model=funnel \
manifold.dim=2 \
manifold.metric=fisher \
manifold.run_evaluation=True \
manifold.make_plots=True
```

To sample the **Two Gaussian Mixture**:
```bash
python magss/sample.py \
rng_key=1 \
sampler=agss \
manifold.model=twogaussians \
manifold.dim=2 \
manifold.metric=inverse_monge \
manifold.metric_param=0.1 \
manifold.run_evaluation=True \
manifold.make_plots=True
```

## Configuration

It is managed by `hydra`
The structure can be found in `conf/manifold/euclidean.yaml` and `conf/sampler/choose_your_sampler.yaml`. The available options inside `euclidean.yaml` are:

- **run_evaluation**: `False` or `True`
  Determines whether to run the evaluation process.

- **make_plots**: `False` or `True`
  Specifies whether to generate plots during execution.

- **num_samples**: integer
  The total number of samples to generate.

- **num_chains**: integer
  The number of chains to run in the sampling process.

- **burnin**: integer
  The number of initial samples to discard as burn-in.

- **thinning**: integer
  The interval for thinning the samples; only every nth sample is retained.

- **model_rng_key**: integer
  The random number generator key for model initialization.

- **dim**: integer
  The dimensionality of the data or model.

- **metric**: options: `"euclidean"`, `"generative"`, `"monge"`, `"fisher"`, `"inverse_generative"`, `"inverse_monge"`
  The metric to be used for measuring distances.

- **metric_param**: float value (≥ 0)
  A parameter associated with the chosen metric.

- **model**: options:
  Specifies the model to be used.

- **integrator**: options: `"euler"`, `"tsit"`, `"dopri5"`, `"dopri8"`, `"kv3"`, `"kv5"`, `"revheun"`
  The numerical integrator to use for solving differential equations.

- **step_size_ode**: `'null'` if adaptive, float value (> 0) if fixed.
  The step size for the ODE solver; can be adaptive or fixed.

`choose_your_sampler.yaml` can be one of the following options:

1. `agss.yaml`
2. `digs.yaml`
3. `mala.yaml`
4. `meta_agss.yaml`
5. `pt.yaml`

Consult each individual `.yaml` file for the specific arguments and configurations.





Major requirements:
 - Python 3.12.1
 - jax                     0.4.38
 - blackjax                1.2.2
 - diffrax                 0.6.2
 - hydra-core              1.3.2
 - POT                     0.9.4
 - numpy                   1.26.4
 - matplotlib              3.10.0
 - pandas                  2.2.2