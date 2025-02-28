import jax.random as jr
import jax.numpy as jnp
from .geodesics import (
    numerical_integrate_geodesic_fn,
    numerical_integrate_geodesic_times_fn,
)
import diffrax

default_solver = diffrax.Dopri5()


class Metric:

    def __init__(
        self,
        dim,
        step_size_ode=None,
        solver=default_solver,
        set_metric_functions=None,
        kwargs={},
    ):
        self.dim = dim
        self.step_size_ode = step_size_ode
        self.solver = solver
        if set_metric_functions is not None:
            (
                self.christoffel_fn,
                self.squared_norm_fn,
                self.inverse_sqrt_metric_fn,
                self.log_determinant_fn,
            ) = set_metric_functions(**kwargs)
        else:
            self.christoffel_fn = lambda x, v: jnp.zeros_like(v)
            self.squared_norm_fn = lambda x, v: jnp.dot(v, v)
            self.inverse_sqrt_metric_fn = lambda x: jnp.eye(dim)
            self.log_determinant_fn = lambda x: 0.0

    def sample_unit_ball(self, rng_key, x):
        v = self.sample_velocity(rng_key, x)
        squared_norm = self.squared_norm_fn(x, v)
        v /= jnp.sqrt(squared_norm)
        return v

    def sample_velocity(self, rng_key, x):
        v = jr.normal(rng_key, x.shape)
        inverse_sqrt_metric = self.inverse_sqrt_metric_fn(x)
        return jnp.dot(inverse_sqrt_metric, v)

    def geodesic_fn(self, x, v, t):
        dim = self.dim
        step_size_ode = self.step_size_ode
        solver = self.solver
        output = numerical_integrate_geodesic_fn(
            dim, self.christoffel_fn, x, v, t, step_size_ode, solver
        )
        x_new = output[-1, :dim]
        v_new = output[-1, dim:]
        return x_new, v_new

    def geodesic_vectorized_fn(self, x, v, ts):
        dim = self.dim
        step_size_ode = self.step_size_ode
        solver = self.solver
        output = numerical_integrate_geodesic_times_fn(
            dim, self.christoffel_fn, x, v, ts, step_size_ode, solver
        )
        x_new = output[:, :dim]
        v_new = output[:, dim:]
        return x_new, v_new
