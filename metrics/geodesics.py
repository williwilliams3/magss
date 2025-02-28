import jax
import jax.numpy as jnp
import diffrax
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
)


def get_step_size_controller(step_size, t0, t1):
    if step_size is not None:
        dt0 = step_size * (t1 - t0)
        return diffrax.ConstantStepSize(), dt0
    else:
        dt0 = None
        return diffrax.PIDController(rtol=1e-5, atol=1e-6), dt0


def univariate_christoffel_fn(logmetric, x, v):
    return 0.5 * jax.grad(logmetric)(x) * v**2


def general_christoffel_fn(g, x, v):
    # adapted based on ChatGPT
    d_g = jax.jacfwd(g)(x)

    # Compute the Christoffel symbols
    partial_1 = jnp.einsum("jli,i,j->l", d_g, v, v)
    partial_2 = jnp.einsum("ilj,i,j->l", d_g, v, v)
    partial_3 = jnp.einsum("ijl,i,j->l", d_g, v, v)
    result = jax.scipy.linalg.solve(
        g(x), 0.5 * (partial_1 + partial_2 - partial_3), assume_a="pos"
    )

    return result


def get_jax_christoffel_fun(dim, christoffel_fn):
    def func(t, y, args):
        theta = y[:dim]
        v = y[dim:]
        a = -christoffel_fn(theta, v)
        return jnp.concatenate([v, a])

    return func


def jax_christoffel_geodesic(dim, christoffel_fn, x, v, t, step_size_ode, solver):

    fun = get_jax_christoffel_fun(dim, christoffel_fn=christoffel_fn)
    term = ODETerm(fun)
    y0 = jnp.concatenate([x, v])
    t0 = 0.0
    t1 = t
    step_size_controller, dt0 = get_step_size_controller(step_size_ode, t0, t1)
    return diffeqsolve(
        term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        stepsize_controller=step_size_controller,
        throw=False,
    )


def jax_christoffel_geodesic_times(
    dim, christoffel_fn, x, v, ts, step_size_ode, solver
):
    # Note if ts[-1] == t0 might cause numerical errors.
    fun = get_jax_christoffel_fun(dim, christoffel_fn=christoffel_fn)
    term = ODETerm(fun)
    y0 = jnp.concatenate([x, v])
    t0 = 0.0
    t1 = ts[-1]
    ts = ts[:-1]
    step_size_controller, dt0 = get_step_size_controller(step_size_ode, t0, t1)
    return diffeqsolve(
        term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        throw=False,
        stepsize_controller=step_size_controller,
        saveat=SaveAt(ts=ts, t1=True),
    )


def numerical_integrate_geodesic_fn(
    dim, christoffel_fn, x, v, t, step_size_ode, solver
):
    sol = jax_christoffel_geodesic(dim, christoffel_fn, x, v, t, step_size_ode, solver)
    output = sol.ys
    # print(sol.stats["num_steps"])
    return output


def numerical_integrate_geodesic_times_fn(
    dim, christoffel_fn, x, v, ts, step_size_ode, solver
):
    sol = jax_christoffel_geodesic_times(
        dim, christoffel_fn, x, v, ts, step_size_ode, solver
    )
    output = sol.ys
    return output
