from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, communicate
from hibayes.communicate.utils import drop_not_present_vars
from typing import Any, Tuple, List

import matplotlib.pyplot as plt
import arviz as az
import arviz.labels as azl
import scienceplots as _scienceplots  # type: ignore[import-not-found]


plt.style.use("science")


VAR_NAME_MAP = {
    "grader_model_id_effects": "Grader",
}

GRADER_LABEL_MAP = {
    "Grader[anthropic/claude-haiku-4.5]": "Claude Haiku 4.5",
    "[google/gemini-2.5-flash]": "Gemini 2.5 Flash",
    "[openai/gpt-5-mini]": "GPT-5 Mini",
}


def _labeller() -> azl.MapLabeller:
    return azl.MapLabeller(var_name_map=VAR_NAME_MAP, coord_map=GRADER_LABEL_MAP)


@communicate
def forest_plot_custom(
    vars: List[str] | None = None,
    vertical_line: float | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (8, 4),
    transform: bool = False,
    *args: Any,
    **kwargs: Any,
):
    def _inner(
        state: AnalysisState,
        display: Any | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        nonlocal vars

        if best_model:
            model_analysis = state.get_best_model()
            assert model_analysis is not None, "No best model found."
            models_to_plot = [model_analysis]
        else:
            models_to_plot = state.models

        for model_analysis in models_to_plot:
            if not model_analysis.is_fitted:
                continue
            vars, dropped = (
                drop_not_present_vars(vars, model_analysis.inference_data)
                if vars
                else (None, None)
            )
            if dropped and display is not None:
                display.logger.warning(
                    f"Variables {dropped} were not found in model {model_analysis.model_name}."
                )
            if vars is None:
                vars = model_analysis.model_config.get_plot_params()

            ax = az.plot_forest(
                model_analysis.inference_data,
                var_names=vars,
                figsize=figsize,
                transform=model_analysis.link_function if transform else None,
                labeller=_labeller(),
                *args,
                **kwargs,
            )
            if vertical_line is not None:
                ax[0].axvline(
                    x=vertical_line,
                    color="red",
                    linestyle="--",
                )
            fig = plt.gcf()
            state.add_plot(
                plot=fig,
                plot_name=f"model_{model_analysis.model_name}_{'-'.join(vars) if vars else ''}_forest",
            )
        return state, "pass"

    return _inner


