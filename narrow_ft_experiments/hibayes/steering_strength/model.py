from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hibayes.process import Features
from hibayes.model.models import Model, model
from hibayes.model.utils import create_interaction_effects
from hibayes.model.models import check_features  # or copy the helper if needed


@model
def normal_main_effects_model(
    main_effects: Optional[List[str]] = None,
    interactions: Optional[List[Tuple[str, str]]] = None,
    effect_coding_for_main_effects: bool = True,
    # priors
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.0,
    prior_sigma_obs_scale: float = 1.0,
    prior_effect_sigma_scale: float = 0.5,
    prior_interaction_loc: float = 0.0,
    prior_interaction_scale: float = 0.3,
) -> Model:
    """
    Hierarchical Normal regression with categorical main effects and optional interactions.

    Outcome: continuous `obs` (e.g. log(max steering strength))
    Predictors: categorical main_effects (e.g. ['grader', 'model', 'position', 'type'])

    For each main effect:
        - sample a group-level scale sigma_effect ~ HalfNormal(prior_effect_sigma_scale)
        - sample per-level effects ~ Normal(0, sigma_effect), with effect-coding if requested

    This matches the style of `linear_group_binomial` and `ordered_logistic_model`,
    but uses a Normal likelihood and hierarchical (partial-pooled) effects.
    """

    def model(features: Features) -> None:
        # ------------------------------------------------------------------
        # 1. Required features
        # ------------------------------------------------------------------
        required_features = ["obs"]

        if main_effects:
            for effect in main_effects:
                required_features.extend([f"{effect}_index", f"num_{effect}"])

        if interactions:
            for var1, var2 in interactions:
                for feat in [
                    f"{var1}_index",
                    f"num_{var1}",
                    f"{var2}_index",
                    f"num_{var2}",
                ]:
                    if feat not in required_features:
                        required_features.append(feat)

        check_features(features, required_features)

        y = features["obs"]

        # ------------------------------------------------------------------
        # 2. Intercept
        # ------------------------------------------------------------------
        prior_intercept = dist.Normal(prior_intercept_loc, prior_intercept_scale)
        intercept = numpyro.sample("intercept", prior_intercept)

        # linear predictor on the same scale as y (e.g. log strength)
        mu = intercept

        # ------------------------------------------------------------------
        # 3. Hierarchical main effects
        # ------------------------------------------------------------------
        if main_effects:
            for effect in main_effects:
                n_levels = features[f"num_{effect}"]
                idx = features[f"{effect}_index"]

                # Group-level scale for this effect family
                sigma_effect = numpyro.sample(
                    f"sigma_{effect}",
                    dist.HalfNormal(prior_effect_sigma_scale),
                )

                # Per-level coefficients with zero-mean prior and shared scale
                if effect_coding_for_main_effects and n_levels > 1:
                    # Effect coding: sum of coefficients = 0
                    free_coefs = numpyro.sample(
                        f"{effect}_effects_constrained",
                        dist.Normal(0.0, sigma_effect).expand([n_levels - 1]),
                    )
                    last_coef = -jnp.sum(free_coefs)
                    coefs = jnp.concatenate([free_coefs, jnp.array([last_coef])])
                    numpyro.deterministic(f"{effect}_effects", coefs)
                else:
                    # Dummy coding (no reference fixed to 0 here; all partial-pooled)
                    coefs = numpyro.sample(
                        f"{effect}_effects",
                        dist.Normal(0.0, sigma_effect).expand([n_levels]),
                    )
                    numpyro.deterministic(f"{effect}_effects", coefs)

                mu = mu + coefs[idx]

        # ------------------------------------------------------------------
        # 4. Interactions (non-hierarchical, like in linear_group_binomial)
        # ------------------------------------------------------------------
        if interactions:
            prior_interaction = dist.Normal(
                prior_interaction_loc, prior_interaction_scale
            )
            for var1, var2 in interactions:
                interaction_matrix = create_interaction_effects(
                    var1, var2, features, prior_interaction
                )
                idx1 = features[f"{var1}_index"]
                idx2 = features[f"{var2}_index"]
                mu = mu + interaction_matrix[idx1, idx2]

        # ------------------------------------------------------------------
        # 5. Observation noise & likelihood
        # ------------------------------------------------------------------
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(prior_sigma_obs_scale))

        numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=y)

    return model
