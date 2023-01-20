import jax
from jax import numpy as jnp


def safe_log(x):
    return jnp.log(jnp.where(x > 0.0, x, 1.0))


def get_latent_type(latent_type: str):
    if latent_type == "gaussian" or latent_type == "tanh_gaussian":
        latent_KL = gaussian_KL
        latent_entropy = gaussian_entropy
        latent_cross_entropy = gaussian_cross_entropy
    elif latent_type == "categorical":
        latent_KL = categorical_KL
        latent_entropy = categorical_entropy
        latent_cross_entropy = categorical_cross_entropy
    else:
        raise ValueError("Unrecognized latent type.")
    return latent_KL, latent_entropy, latent_cross_entropy


########################################################################
# Probability Helper Functions
########################################################################
def log_gaussian_probability(x, params):
    mu = params["mu"]
    sigma = params["sigma"]
    return -(
        safe_log(sigma) + 0.5 * safe_log(2 * jnp.pi) + 0.5 * ((x - mu) / sigma) ** 2
    )


def gaussian_cross_entropy(params_1, params_2):
    mu_1 = params_1["mu"]
    sigma_1 = params_1["sigma"]
    mu_2 = params_2["mu"]
    sigma_2 = params_2["sigma"]
    return (
        0.5 * safe_log(2 * jnp.pi)
        + safe_log(sigma_2)
        + (sigma_1**2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2**2)
    )


def gaussian_entropy(params):
    sigma = params["sigma"]
    return 0.5 + 0.5 * safe_log(2 * jnp.pi) + safe_log(sigma)


def gaussian_KL(params_1, params_2):
    return gaussian_cross_entropy(params_1, params_2) - gaussian_entropy(params_1)


def log_binary_probability(x, params):
    logit = params["logit"]
    # return jnp.where(x, jax.nn.log_sigmoid(logit), jax.nn.log_sigmoid(-logit))
    return jax.nn.log_sigmoid(jnp.where(x, logit, -logit))


def binary_entropy(params):
    logit = params["logit"]
    return jax.nn.sigmoid(logit) * jax.nn.log_sigmoid(logit) + jax.nn.sigmoid(
        -logit
    ) * jax.nn.log_sigmoid(-logit)


def categorical_cross_entropy(params_1, params_2):
    probs_1 = params_1["probs"]
    log_probs_2 = params_2["log_probs"]
    return -jnp.sum(probs_1 * log_probs_2, axis=(-1))


def categorical_entropy(params):
    probs = params["probs"]
    log_probs = params["log_probs"]
    return -jnp.sum(probs * log_probs, axis=(-1))


def categorical_KL(params_1, params_2):
    return categorical_cross_entropy(params_1, params_2) - categorical_entropy(params_1)
