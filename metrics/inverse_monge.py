import jax
import jax.numpy as jnp
from .metrics import Metric


class InverseMonge(Metric):

    def __init__(self, dim, step_size_ode, solver, kwargs):

        super().__init__(
            dim,
            step_size_ode,
            solver,
            set_metric_functions=set_inverse_monge,
            kwargs=kwargs,
        )


def set_inverse_monge(logdensity_fn, alpha2):
    grad_fn = jax.grad(logdensity_fn)
    jax_grad_and_hvp_fn = lambda x, v: jax.jvp(grad_fn, [x], [v])

    def f_grad_fn(x):
        grad = grad_fn(x)
        _, hvp_grad = jax_grad_and_hvp_fn(x, grad)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        return 2 * alpha2 / (L_alpha**2) * hvp_grad

    def christoffel_fn(x, v):
        f_grad = f_grad_fn(x)
        grad, hvp = jax_grad_and_hvp_fn(x, v)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        f = -1 / L_alpha
        dot_grad_ell_and_v = jnp.dot(grad, v)
        term1 = (
            2
            * L_alpha
            * (jnp.dot(v, f_grad) * dot_grad_ell_and_v + f * jnp.dot(hvp, v))
        )
        term2 = -alpha2 * jnp.dot(grad, f_grad) * dot_grad_ell_and_v**2
        term3 = -(dot_grad_ell_and_v**2)
        return 0.5 * alpha2 * ((term1 + term2) * grad + term3 * f_grad)

    def squared_norm_fn(x, v):
        grad = grad_fn(x)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        f = -1 / L_alpha
        return jnp.square(v).sum() + f * alpha2 * jnp.dot(grad, v) ** 2

    def inverse_sqrt_metric_fn(x):
        grad = grad_fn(x)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        factor = alpha2 / (L_alpha**0.5 + 1.0)
        return jnp.eye(x.shape[0]) + factor * jnp.outer(grad, grad)

    def log_determinant_fn(x):
        grad = grad_fn(x)
        L_alpha = 1 + alpha2 * jnp.dot(grad, grad)
        return -jnp.log(L_alpha)

    return christoffel_fn, squared_norm_fn, inverse_sqrt_metric_fn, log_determinant_fn
