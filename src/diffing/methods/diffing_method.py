from abc import ABC, abstractmethod
from typing import Dict, Optional
from omegaconf import DictConfig
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from src.utils.model import load_model_from_config, load_tokenizer_from_config
from src.utils.configs import get_model_configurations


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
        self._base_model: Optional[AutoModelForCausalLM] = None
        self._finetuned_model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.method_cfg = cfg.diffing.method

    @property
    def base_model(self) -> AutoModelForCausalLM:
        """Load and return the base model."""
        if self._base_model is None:
            self._base_model, self._tokenizer = load_model_from_config(
                self.base_model_cfg
            )
            self._base_model.eval()
        return self._base_model

    @property
    def finetuned_model(self) -> AutoModelForCausalLM:
        """Load and return the finetuned model."""
        if self._finetuned_model is None:
            self._finetuned_model, _ = load_model_from_config(self.finetuned_model_cfg)
            self._finetuned_model.eval()
        return self._finetuned_model

    def clear_base_model(self) -> None:
        """Clear the base model from memory."""
        del self._base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self._base_model = None
        logger.info("Cleared base model from CUDA memory with garbage collection")

    def clear_finetuned_model(self) -> None:
        """Clear the finetuned model from memory."""
        del self._finetuned_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self._finetuned_model = None
        logger.info("Cleared finetuned model from CUDA memory with garbage collection")
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Load and return the tokenizer from the base model."""
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer_from_config(self.base_model_cfg)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def setup_models(self) -> None:
        """Ensure both models and tokenizer are loaded."""
        _ = self.base_model  # Triggers loading
        _ = self.finetuned_model  # Triggers loading
        self.logger.info("Models loaded successfully")

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

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Move model to device if needed
        model = model.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                disable_compile=True # TODO: figure out why compiling this crashes the model
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated_text

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
