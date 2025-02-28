import jax
import jax.numpy as jnp
from .metrics import Metric


class Monge(Metric):

    def __init__(self, dim, step_size_ode, solver, kwargs):
        super().__init__(
            dim, step_size_ode, solver, set_metric_functions=set_monge, kwargs=kwargs
        )


def set_monge(logdensity_fn, alpha2):
    grad_fn = jax.grad(logdensity_fn)

    def christoffel_fn(x, v):
        jax_grad_and_hvp_fn = lambda x, v: jax.jvp(grad_fn, [x], [v])
        theta_grad, theta_hvp_v = jax_grad_and_hvp_fn(x, v)
        norm_theta_grad_2 = jnp.dot(theta_grad, theta_grad)

        W_2 = 1.0 + alpha2 * norm_theta_grad_2
        mho = alpha2 * jnp.dot(v, theta_hvp_v) / W_2
        return mho * theta_grad

    def squared_norm_fn(x, v):
        return jnp.square(v).sum() + alpha2 * jnp.dot(grad_fn(x), v) ** 2

    def inverse_sqrt_metric_fn(x):
        grad = grad_fn(x)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        factor = -alpha2 / (L_alpha**0.5 + L_alpha)
        return jnp.eye(x.shape[0]) + factor * jnp.outer(grad, grad)

    def log_determinant_fn(x):
        grad = grad_fn(x)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        return jnp.log(L_alpha)

    return christoffel_fn, squared_norm_fn, inverse_sqrt_metric_fn, log_determinant_fn
