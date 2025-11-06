from abc import ABC, abstractmethod
from typing import Dict, List
from omegaconf import DictConfig
from pathlib import Path
import torch as th

from loguru import logger
from nnterp import StandardizedTransformer


from src.utils.model import (
    load_model_from_config,
    # place_inputs,
    gc_collect_cuda_cache,
    AnyTokenizer,
)
from src.utils.configs import get_model_configurations
from src.utils.agents.base_agent import BaseAgent
from src.utils.agents.blackbox_agent import BlackboxAgent
from src.utils.agents.diffing_method_agent import DiffingMethodAgent


class DiffingMethod(ABC):
    """
    Abstract base class for diffing methods.

    Handles common functionality like model loading, tokenizer access,
    and configuration management.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logger.bind(method=self.__class__.__name__)

        # Extract model configurations
        self.base_model_cfg, self.finetuned_model_cfg = get_model_configurations(cfg)

        # Initialize model and tokenizer placeholders
        self._base_model: StandardizedTransformer | None = None
        self._finetuned_model: StandardizedTransformer | None = None
        self._tokenizer: AnyTokenizer | None = None

        # Set device
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.method_cfg = cfg.diffing.method

    @property
    def base_model(self) -> StandardizedTransformer:
        """Load and return the base model."""
        if self._base_model is None:
            self._base_model = load_model_from_config(self.base_model_cfg)
            self._base_model.eval()
        return self._base_model

    @property
    def finetuned_model(self) -> StandardizedTransformer:
        """Load and return the finetuned model."""
        if self._finetuned_model is None:
            self._finetuned_model = load_model_from_config(self.finetuned_model_cfg)
            self._finetuned_model.eval()
        return self._finetuned_model

    def clear_base_model(self) -> None:
        """Clear the base model from memory."""
        del self._base_model
        gc_collect_cuda_cache()
        self._base_model = None
        logger.info("Cleared base model from CUDA memory with garbage collection")

    def clear_finetuned_model(self) -> None:
        """Clear the finetuned model from memory."""
        del self._finetuned_model
        gc_collect_cuda_cache()
        self._finetuned_model = None
        logger.info("Cleared finetuned model from CUDA memory with garbage collection")

    @property
    def tokenizer(self) -> AnyTokenizer:
        """Load and return the tokenizer from the base model."""
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            if self._tokenizer is None:
                self._tokenizer = self.finetuned_model.tokenizer
                if self._tokenizer.pad_token is None:
                    raise ValueError(
                        "Clement: Unexpected: nnsight / utils.model should have set the pad token"
                    )
                    # self._tokenizer.pad_token = self._tokenizer.eos_token

                # Check if tokenizer has chat template
                if self._tokenizer.chat_template is None:
                    logger.warning(
                        "Tokenizer does not have chat template. Using base model tokenizer"
                    )
                    raise ValueError(
                        "Finetuned model tokenizer does not have chat template"
                    )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}. Retrying with base model...")
            self._tokenizer = self.base_model.tokenizer
            if self._tokenizer.pad_token is None:
                raise ValueError(
                    "Clement: Unexpected: nnsight / utils.model should have set the pad token"
                )
                # self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def setup_models(self) -> None:
        """Ensure both models and tokenizer are loaded."""
        _ = self.base_model  # Triggers loading
        _ = self.finetuned_model  # Triggers loading
        self.logger.info("Models loaded successfully")

    @th.no_grad()
    def generate_text(
        self,
        prompt: str,
        model_type: str = "base",
        max_length: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text using either the base or finetuned model.

        Args:
            prompt: Input prompt text
            model_type: Either "base" or "finetuned"
            max_length: Maximum length of generated text
            temperature: Sampling temperature

        Returns:
            Generated text (including the original prompt)
        """
        import streamlit as st

        # Select the appropriate model
        if model_type == "base":
            with st.spinner("Loading base model..."):
                model = self.base_model
        elif model_type == "finetuned":
            with st.spinner("Loading finetuned model..."):
                model = self.finetuned_model
        else:
            raise ValueError(
                f"model_type must be 'base' or 'finetuned', got: {model_type}"
            )

        # Tokenize input and place for the selected model
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        # placed = place_inputs(inputs["input_ids"], inputs["attention_mask"], model)
        placed = inputs  # TODO: clean
        input_ids = placed["input_ids"]
        attention_mask = placed["attention_mask"]

        # Generate
        with model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + max_length,
            temperature=temperature,
            do_sample=do_sample,
        ):
            outputs = model.generator.output.save()

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated_text

    @th.no_grad()
    def generate_texts(
        self,
        prompts: List[str],
        model_type: str = "base",
        max_length: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_only_generation: bool = False,
    ) -> List[str]:
        """Batch generate texts using either the base or finetuned model.

        Args:
            prompts: List of input prompt texts
            model_type: Either "base" or "finetuned"
            max_length: Maximum number of tokens to generate beyond the input length
            temperature: Sampling temperature
            do_sample: Whether to sample
            return_only_generation: If True, return only the generated continuation
                after the input prompt for each example (decoded with special tokens skipped).

        Returns:
            List of generated texts (each includes its original prompt)
        """
        import streamlit as st

        assert (
            isinstance(prompts, list)
            and len(prompts) > 0
            and all(isinstance(p, str) and len(p) > 0 for p in prompts)
        )

        if model_type == "base":
            with st.spinner("Loading base model..."):
                model = self.base_model
        elif model_type == "finetuned":
            with st.spinner("Loading finetuned model..."):
                model = self.finetuned_model
        else:
            raise ValueError(
                f"model_type must be 'base' or 'finetuned', got: {model_type}"
            )

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # placed = place_inputs(input_ids, attention_mask, model)
        # input_ids = placed["input_ids"]
        # attention_mask = placed["attention_mask"]
        assert input_ids.ndim == 2 and attention_mask.ndim == 2
        assert input_ids.shape == attention_mask.shape

        base_len = input_ids.shape[1]
        with model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=base_len + max_length,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            disable_compile=True,
        ):
            outputs = model.generator.output.save()
        if return_only_generation:
            # Slice off the input portion per-example using true input lengths
            input_lengths: List[int] = attention_mask.sum(dim=1).tolist()
            continuations: List[str] = []
            for i, inp_len in enumerate(input_lengths):
                # Guard against pathological cases
                assert isinstance(inp_len, int) and inp_len >= 0
                gen_ids = outputs[i, int(inp_len) :].tolist()
                continuations.append(
                    self.tokenizer.decode(
                        gen_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            return continuations
        else:
            decoded: List[str] = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=False
            )
            return decoded

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def visualize(self):
        pass

    @staticmethod
    @abstractmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available results for this method.

        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return getattr(self.cfg, "verbose", False)

    # Agent methods
    @abstractmethod
    def get_agent(self) -> DiffingMethodAgent:
        """Get the agent for the method."""
        raise NotImplementedError

    def get_baseline_agent(self) -> BlackboxAgent:
        return BlackboxAgent(cfg=self.cfg)
