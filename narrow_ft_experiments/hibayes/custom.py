from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult
from hibayes.communicate.utils import drop_not_present_vars
import matplotlib.pyplot as plt
from typing import Tuple, Any
import arviz as az
import arviz.labels as azl
import scienceplots as _scienceplots  # type: ignore[import-not-found]

plt.style.use("science")


from hibayes.communicate import communicate

MAP = {
    "ADL_effects": "ADL ",
    "interactions_effects": "Interactions ",
    "organism_type_effects": "Organism Type ",
    "model_effects": "Model ",
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
    "qwen25_VL_3B_Instruct": "Qwen2.5 VL 3B",
}
labeller = azl.MapLabeller(var_name_map=MAP)


@communicate
def forest_plot_custom(
    vars: list[str] | None = None,
    vertical_line: float | None = None,
    best_model: bool = True,
    figsize: tuple[int, int] = (10, 10),
    transform: bool = False,
    *args,
    **kwargs,
):
    def communicate(
        state: AnalysisState,
        display: Any | None = None,
    ) -> Tuple[AnalysisState, CommunicateResult]:
        """
        Communicate the results of a model analysis.
        """
        nonlocal vars
        if best_model:
            best_model_analysis = state.get_best_model()
            if best_model_analysis is None:
                raise ValueError("No best model found.")
            models_to_run = [best_model_analysis]
        else:
            models_to_run = state.models

        for model_analysis in models_to_run:
            if model_analysis.is_fitted:
                vars, dropped = (
                    drop_not_present_vars(vars, model_analysis.inference_data)
                    if vars
                    else (None, None)
                )
                if dropped and display:
                    display.logger.warning(
                        f"Variables {dropped} were not found in the model {model_analysis.model_name} inference data."
                    )
                if vars is None:
                    vars = model_analysis.model_config.get_plot_params()

                ax = az.plot_forest(
                    model_analysis.inference_data,
                    var_names=vars,
                    figsize=figsize,
                    transform=model_analysis.link_function if transform else None,
                    labeller=labeller,
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

    return communicate
