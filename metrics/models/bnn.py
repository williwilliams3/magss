import jax.numpy as jnp
import flax.linen as nn
import jax.random as jr
from jax.nn import relu


# Define the MLP with one hidden layer and ReLU activation
class MLP(nn.Module):
    d_x: int
    d_h: int

    def setup(self):
        self.w1 = self.param("w1", nn.initializers.normal(), (self.d_x, self.d_h))
        self.b1 = self.param("b1", nn.initializers.normal(), (self.d_h,))
        self.w2 = self.param("w2", nn.initializers.normal(), (self.d_h, 1))

    def __call__(self, x):
        z1 = jnp.dot(x, self.w1) / jnp.sqrt(self.d_x) + self.b1 * 0.1
        a1 = relu(z1)
        y = jnp.dot(a1, self.w2) / jnp.sqrt(self.d_h)
        return y[..., 0]


def get_param_dim(d_x, d_h):
    return d_x * d_h + d_h + d_h


def reshape_params(params, d_x, d_h):
    return {
        "params": {
            "w1": params[: d_x * d_h].reshape(d_x, d_h),
            "b1": params[d_x * d_h : (d_x + 1) * d_h].reshape(
                d_h,
            ),
            "w2": params[(d_x + 1) * d_h :].reshape(d_h, 1),
        }
    }


def get_ground_truth_bnn(key, mlp, d_x, s_n):

    param_key, x_train_key, x_test_key = jr.split(key, 3)
    x_train = jr.normal(x_train_key, (500, d_x))
    # Initialize mlp
    param_true = mlp.init(param_key, x_train)

    y_train = mlp.apply(param_true, x_train)
    y_train = y_train + s_n * jr.normal(x_train_key, y_train.shape)

    x_test = jr.normal(x_test_key, (500, d_x))
    y_test = mlp.apply(param_true, x_test)
    y_test = y_test + s_n * jr.normal(x_test_key, y_test.shape)

    return x_train, x_test, y_train, y_test, param_true


class BNN:

    def __init__(
        self,
        rng_key,
        d_x=8,
        d_h=10,
        s_n=0.1,
    ):
        self.d_x = d_x
        self.d_h = d_h
        self.s_n = s_n
        mlp = MLP(d_x, d_h)
        self.mlp = mlp
        self.dim = get_param_dim(d_x, d_h)
        self.x_train, self.x_test, self.y_train, self.y_test, self.param_true = (
            get_ground_truth_bnn(rng_key, mlp, d_x, s_n)
        )
        self.name = "BNN"

    def logdensity_fn(self, x):
        x_train = self.x_train
        y_train = self.y_train
        mlp = self.mlp

        prior_value = -0.5 * jnp.sum(x**2)
        x = reshape_params(x, self.d_x, self.d_h)
        y_train_pred = mlp.apply(x, x_train)
        loglikelihood_value = (
            -0.5 * jnp.sum((y_train_pred - y_train) ** 2) / (self.s_n**2)
        )
        return prior_value + loglikelihood_value

    def sample(self):
        raise NotImplementedError("Sampling not implemented for BNN")

    def evaluate_test_nll(self, params):
        s_n = self.s_n
        x_test = self.x_test
        y_test = self.y_test
        mlp = self.mlp
        if not isinstance(params, dict):
            params = reshape_params(params, self.d_x, self.d_h)

        y_test_pred = mlp.apply(params, x_test)
        nll = 0.5 * jnp.log(2.0 * jnp.pi * s_n**2) + 0.5 * jnp.mean(
            (y_test_pred - y_test) ** 2
        ) / (s_n**2)
        return nll


if __name__ == "__main__":
    # Example usage:
    rng_key = jr.key(0)
    model = BNN(rng_key)
    nll = model.evaluate_test_nll(model.param_true)
    print("NLL on test data:", nll)
    print("Total dimensions:", model.dim)

    # nll on random parameters
    params = jr.normal(rng_key, (model.dim,))
    nll = model.evaluate_test_nll(params)
    print("NLL on test data for random parameters:", nll)
