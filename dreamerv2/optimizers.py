import haiku as hk
import optax


def apply_updates(
    optimizer: optax.GradientTransformation,
    params: hk.Params,
    opt_state: optax.OptState,
    grads,
):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
