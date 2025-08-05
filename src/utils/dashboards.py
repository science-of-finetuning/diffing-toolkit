"""
Base dashboard utilities for interactive diffing method analysis.
Follows the pattern of AbstractOnlineFeatureCentricDashboard from tiny_dashboard.
"""

import time
import traceback
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st
import torch
import numpy as np
from IPython.display import HTML
import sqlite3

# Import tiny_dashboard components
from tiny_dashboard.html_utils import (
    create_example_html,
    create_base_html,
    create_highlighted_tokens_html,
)
from tiny_dashboard.html_utils import (
    styles as default_styles,
    scripts as default_scripts,
)
from tiny_dashboard.utils import apply_chat
from src.utils.visualization import (
    filter_examples_by_search,
    create_examples_html,
    render_streamlit_html,
)
from src.utils.model import has_thinking

class AbstractOnlineDiffingDashboard(ABC):
    """
    Abstract base class for real-time diffing analysis dashboards.
    Users can input text and see the per-token differences highlighted directly in the text.
    Closely follows AbstractOnlineFeatureCentricDashboard pattern.
    """

    def __init__(self, method_instance):
        """
        Args:
            method_instance: Instance of a DiffingMethod (e.g., KLDivergenceDiffingMethod)
        """
        self.method = method_instance
        self.use_chat_formatting = False
        self.current_html = None

    def prepare_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text using the method's tokenizer.

        Args:
            text: Input text to tokenize

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        return self.method.tokenizer(text, return_tensors="pt", add_special_tokens=True)

    def create_token_visualization(
        self, tokens: list, values: np.ndarray, color_rgb: tuple = (255, 0, 0)
    ) -> str:
        """
        Create HTML visualization with color-coded tokens using tiny_dashboard.

        Args:
            tokens: List of token strings
            values: Numpy array of values per token
            color_rgb: RGB color tuple for highlighting

        Returns:
            HTML string with color-coded tokens
        """
        if len(values) == 0:
            return "<p>No tokens to display</p>"

        # Convert to torch tensor for tiny_dashboard compatibility
        activations = torch.tensor(values).unsqueeze(1)  # Shape: [seq_len, 1]

        # Create highlighted tokens HTML (just the spans)
        tokens_html = create_highlighted_tokens_html(
            tokens=tokens,
            activations=activations,
            tokenizer=self.method.tokenizer,
            highlight_features=[0],  # Single feature case
            color1=color_rgb,
            activation_names=["Activation"],
        )

        # Get max value for display
        max_value = float(np.max(values)) if len(values) > 0 else 0.0

        # Ensure tokens_html is a string (handle potential tuple return)
        if isinstance(tokens_html, tuple):
            tokens_html = tokens_html[0]  # Take first element if tuple

        # Wrap in example container with borders and styling (same as KL examples)
        # For static display, we need to ensure the copy function works by setting up full_html properly
        example_html = create_example_html(
            max_act=f"{max_value:.3f}",
            collapsed_html=tokens_html,
            full_html=tokens_html,  # Same content for both collapsed and full
            static=True,  # No expand/collapse functionality needed
        )

        custom_styles = (
            default_styles
            + """
        .token {
            font-size: 22px !important;
            line-height: 1.4 !important;
        }
        .example-container {
            font-size: 14px;
        }
        """
        )

        # Enhanced scripts to fix copy functionality for static examples
        enhanced_scripts = (
            default_scripts
            + """
        // Fix copy functionality for static examples
        window.copyExampleToClipboard = function (event, container) {
            event.preventDefault(); // Prevent default context menu

            // For static examples, get tokens from the visible collapsed-text div
            const isStatic = container.dataset.static === 'true';
            let tokensContainer;
            
            if (isStatic) {
                tokensContainer = container.querySelector('.collapsed-text');
            } else {
                tokensContainer = container.querySelector('.full-text');
            }
            
            if (!tokensContainer) {
                console.warn('No tokens container found');
                return;
            }

            const textToCopy = Array.from(tokensContainer.querySelectorAll('.token'))
                .map(token => {
                    return token.dataset.tokenstr;
                })
                .join('');
            
            console.log("Attempting to copy text: ", textToCopy);

            // Function to show visual feedback
            function showCopiedFeedback() {
                const sampleContainer = container.closest('.text-sample');
                sampleContainer.classList.add('copied');
                setTimeout(() => {
                    sampleContainer.classList.remove('copied');
                }, 1000);
            }

            // Try multiple copy methods
            function tryFallbackCopy() {
                console.log('Trying fallback copy method...');
                try {
                    const textArea = document.createElement('textarea');
                    textArea.value = textToCopy;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-999999px';
                    textArea.style.top = '-999999px';
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    
                    const successful = document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    if (successful) {
                        console.log('Fallback copy successful');
                        showCopiedFeedback();
                        return true;
                    } else {
                        console.error('Fallback copy failed');
                        return false;
                    }
                } catch (fallbackErr) {
                    console.error('Fallback copy exception: ', fallbackErr);
                    return false;
                }
            }

            // Check if modern clipboard API is available and try it first
            if (navigator.clipboard && navigator.clipboard.writeText) {
                console.log('Trying modern clipboard API...');
                navigator.clipboard.writeText(textToCopy).then(() => {
                    console.log('Modern clipboard API successful');
                    showCopiedFeedback();
                }).catch(err => {
                    console.error('Modern clipboard API failed: ', err);
                    // Try fallback method
                    if (!tryFallbackCopy()) {
                        // If all else fails, still show feedback but log the failure
                        console.error('All copy methods failed');
                        showCopiedFeedback(); // Show feedback anyway for UX
                    }
                });
            } else {
                console.log('Modern clipboard API not available, using fallback...');
                // Try fallback method directly
                if (!tryFallbackCopy()) {
                    console.error('All copy methods failed');
                    showCopiedFeedback(); // Show feedback anyway for UX
                }
            }
        };
        """
        )

        # Wrap in complete HTML with CSS using create_base_html
        complete_html = create_base_html(
            title="Per-Token Analysis",
            content=example_html,
            styles=custom_styles,  # Combine default and custom styles
            scripts=enhanced_scripts,  # Use enhanced scripts with fixed copy functionality
        )

        return complete_html

    @abstractmethod
    def compute_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute statistics for tokenized input by calling the method's computation function.

        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor

        Returns:
            Dictionary with computed statistics
        """
        pass

    @abstractmethod
    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get method-specific parameters (e.g., selected layer for NormDiff)."""
        pass

    def _get_color_rgb(self) -> tuple:
        """Get red color for divergence highlighting."""
        return (255, 0, 0)

    @abstractmethod
    def _get_title(self) -> str:
        """Get the title for the analysis."""
        pass

    def display(self):
        """Render interface with integrated analysis and generation."""
        st.markdown(f"### {self._get_title()}")
        st.markdown(
            "Enter text to see per-token differences between base and finetuned models."
        )

        # Create method-specific session state keys
        method_key = self._get_title().replace(' ', '_').lower()
        session_keys = {
            'use_chat': f"{method_key}_use_chat",
            'enable_generation': f"{method_key}_enable_generation",
            'model_type': f"{method_key}_model_type",
            'max_length': f"{method_key}_max_length",
            'temperature': f"{method_key}_temperature",
            'do_sample': f"{method_key}_do_sample",
            'text_input': f"{method_key}_text_input",
            'analysis_results': f"analysis_results_{method_key}",
        }

        # Initialize session state with defaults
        if session_keys['use_chat'] not in st.session_state:
            st.session_state[session_keys['use_chat']] = True
        if session_keys['enable_generation'] not in st.session_state:
            st.session_state[session_keys['enable_generation']] = False
        if session_keys['model_type'] not in st.session_state:
            st.session_state[session_keys['model_type']] = "base"
        if session_keys['max_length'] not in st.session_state:
            st.session_state[session_keys['max_length']] = 100
        if session_keys['temperature'] not in st.session_state:
            st.session_state[session_keys['temperature']] = 1.0
        if session_keys['do_sample'] not in st.session_state:
            st.session_state[session_keys['do_sample']] = True
        if session_keys['text_input'] not in st.session_state:
            st.session_state[session_keys['text_input']] = ""
        if session_keys['analysis_results'] not in st.session_state:
            st.session_state[session_keys['analysis_results']] = None

        # Chat formatting option
        use_chat = st.checkbox(
            "Use Chat Formatting (add <eot> to switch the user/assistant turn)",
            key=session_keys['use_chat']
        )

        # Generation options (expandable section)
        with st.expander("🤖 Text Generation Options", expanded=False):
            enable_generation = st.checkbox(
                "Enable Text Generation", 
                help="Generate text first, then analyze it",
                key=session_keys['enable_generation']
            )

            if enable_generation:
                col1, col2 = st.columns(2)

                with col1:
                    model_type = st.selectbox(
                        "Generation Model:",
                        options=["base", "finetuned"],
                        help="Choose which model to use for generation",
                        key=session_keys['model_type']
                    )

                with col2:
                    max_length = st.slider(
                        "Max Generation Length:",
                        min_value=10,
                        max_value=500,
                        value=st.session_state[session_keys['max_length']],
                        help="Maximum number of tokens to generate",
                        key=session_keys['max_length']
                    )

                col3, col4 = st.columns(2)

                with col3:
                    temperature = st.slider(
                        "Temperature:",
                        min_value=0.1,
                        max_value=2.0,
                        value=st.session_state[session_keys['temperature']],
                        step=0.1,
                        help="Sampling temperature (lower = more deterministic, higher = more creative)",
                        key=session_keys['temperature']
                    )

                with col4:
                    do_sample = st.checkbox(
                        "Use Sampling",
                        help="Enable sampling (if disabled, uses greedy decoding)",
                        key=session_keys['do_sample']
                    )
                    if has_thinking(self.method.cfg):
                        enable_thinking = st.checkbox(
                            "Enable Thinking",
                            value=False,
                            help="Enable thinking (if disabled, prefills <think> </think> tokens)"
                        )
                    else:
                        enable_thinking = False
            else:
                enable_thinking = False

        # Method-specific controls
        method_params = self._render_streamlit_method_controls()

        # Text input - changes based on generation mode
        if enable_generation:
            text = st.text_area(
                "Prompt for Generation:",
                height=100,
                help="Enter a prompt - the model will generate text and then analyze the full result",
                key=session_keys['text_input']
            )
            analyze_button = st.button(
                "✨ Generate & Analyze", type="primary", use_container_width=True
            )
        else:
            text = st.text_area(
                "Input Text:",
                height=100,
                help="Enter text to analyze differences per token",
                key=session_keys['text_input']
            )
            analyze_button = st.button(
                "🔍 Analyze Text", type="primary", use_container_width=True
            )

        # Process when button is clicked
        if analyze_button:
            # Ensure text is not None and strip whitespace
            current_text = st.session_state[session_keys['text_input']] or ""
            if not current_text.strip():
                if enable_generation:
                    st.warning("Please enter a prompt for generation.")
                else:
                    st.warning("Please enter some text to analyze.")
                return

            # Set chat formatting
            if use_chat:
                current_text = apply_chat(current_text, self.method.tokenizer, add_bos=False, enable_thinking=enable_thinking)

            # Generate text if generation is enabled
            if enable_generation:
                with st.spinner(f"Generating text with {model_type} model..."):
                    try:
                        generated_text = self.method.generate_text(
                            prompt=current_text,
                            model_type=model_type,
                            max_length=max_length,
                            temperature=temperature,
                            do_sample=do_sample,
                        )

                        # Show the generated text
                        st.markdown("### Generated Text")
                        st.info(
                            f"**Model:** {model_type.title()} | **Temperature:** {temperature} | **Max Length:** {max_length} | **Do Sample:** {do_sample}"
                        )
                        st.code(generated_text, language="text")

                        # Use generated text for analysis
                        text_to_analyze = generated_text

                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        return
            else:
                text_to_analyze = current_text

            # Tokenize and analyze the text
            tokens_dict = self.prepare_text(text_to_analyze)
            input_ids = tokens_dict["input_ids"]
            attention_mask = tokens_dict["attention_mask"]

            # Compute statistics
            with st.spinner("Computing differences..."):
                results = self.compute_statistics_for_tokens(
                    input_ids, attention_mask, **method_params
                )

                # Create token visualization
                html_viz = self.create_token_visualization(
                    results["tokens"],
                    results["values"],
                    color_rgb=self._get_color_rgb(),
                )

                # Store results in session state
                st.session_state[session_keys['analysis_results']] = {
                    "total_tokens": results["total_tokens"],
                    "html_viz": html_viz,
                    "generated": enable_generation,
                    "model_type": model_type if enable_generation else None,
                    "original_prompt": current_text if enable_generation else None,
                }

        # Display results if they exist
        if st.session_state[session_keys['analysis_results']] is not None:
            results_data = st.session_state[session_keys['analysis_results']]

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)


            # Show context for generated text
            if results_data.get("generated", False):
                st.markdown(
                    f"*Analysis of text generated by {results_data['model_type']} model from prompt: \"{results_data['original_prompt'][:50]}...\"*"
                )

            # Display visualization
            st.markdown(f"### Per-Token {self._get_title()}")
            st.markdown(
                "*Tokens are colored by their difference values (higher = more intense)*"
            )
            render_streamlit_html(results_data["html_viz"])
        else:
            if enable_generation:
                st.info(
                    "👆 Enter a prompt above and click 'Generate & Analyze' to create text and see per-token differences."
                )
            else:
                st.info(
                    "👆 Enter text above and click 'Analyze Text' to see per-token differences."
                )

    @abstractmethod
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render method-specific controls in Streamlit and return parameters."""
        pass

class SteeringDashboard:
    """
    Base class for steering latent activations during text generation.
    
    This dashboard provides a clean interface for comparing baseline vs steered generation
    without token-wise analysis - just side-by-side text comparison.
    """
    
    def __init__(self, method_instance):
        self.method = method_instance
    
    @property
    @abstractmethod 
    def layer(self) -> int:
        """Get the layer number for this steering dashboard."""
        pass
    
    @abstractmethod
    def get_latent(self, idx: int) -> torch.Tensor:
        """
        Get latent vector for steering.
        
        Args:
            idx: Latent index
            
        Returns:
            Latent vector [hidden_dim] for the specified latent
        """
        pass
    
    @abstractmethod
    def get_dict_size(self) -> int:
        """Get the dictionary size for validation."""
        pass
    
    @abstractmethod
    def _get_title(self) -> str:
        """Get title for steering analysis."""
        pass
    
    @abstractmethod
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render method-specific steering controls in Streamlit and return parameters."""
        pass
    
    def generate_with_steering(
        self, 
        prompt: str, 
        latent_idx: int, 
        steering_factor: float,
        steering_mode: str,
        model_type: str = "base",
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        linear_decay_steps: int = 10,
    ) -> str:
        """
        Generate text with latent steering using nnsight.
        
        Args:
            prompt: Input prompt text
            latent_idx: Latent index to steer
            steering_factor: Strength of steering
            steering_mode: "prompt_only" or "all_tokens" or "linear_decay"
            model_type: "base" or "finetuned"
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
                            
        Returns:
            Generated text with steering applied
        """
        from nnsight import LanguageModel
        
        # Select the appropriate model
        if model_type == "base":
            model = self.method.base_model
        elif model_type == "finetuned":
            model = self.method.finetuned_model
        else:
            raise ValueError(f"model_type must be 'base' or 'finetuned', got: {model_type}")
        
        # Get the latent vector for steering
        latent_vector = self.get_latent(latent_idx)  # [hidden_dim]
        latent_vector = latent_vector.to(self.method.device)
        
        # Create LanguageModel wrapper
        nn_model = LanguageModel(model, tokenizer=self.method.tokenizer)
        
        # Tokenize prompt
        inputs = self.method.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.method.device)
        
        # Shape assertions
        assert input_ids.ndim == 2, f"Expected 2D input_ids, got shape {input_ids.shape}"
        prompt_length = input_ids.shape[1]
        
        if steering_mode == "linear_decay":
            steering_factor_per_token = torch.zeros(linear_decay_steps)
            for i in range(linear_decay_steps):
                steering_factor_per_token[i] = (linear_decay_steps - i) * (steering_factor / float(linear_decay_steps))
            steering_factor = steering_factor_per_token.to(self.method.device)

        # Generate with steering intervention
        with nn_model.generate(
            input_ids, 
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.method.tokenizer.eos_token_id,
            disable_compile=True, # TODO: fix this once nnsight is fixed
        ) as tracer:
            
            if steering_mode == "all_tokens":
                # Apply steering to all tokens (prompt + generated)
                with nn_model.model.layers[self.layer].all():
                    # Add steering vector to layer output
                    # Shape: layer output is [batch_size, seq_len, hidden_dim]
                    # latent_vector is [hidden_dim]
                    # Broadcasting will add the latent_vector to each token position
                    nn_model.model.layers[self.layer].output[0][:] += steering_factor * latent_vector
            elif steering_mode == "linear_decay":
                # Apply steering to all tokens (prompt + generated)
                for i in range(linear_decay_steps):
                    if i == 0:
                        nn_model.model.layers[self.layer].output[0][:] += steering_factor_per_token[i] * latent_vector
                    else:
                        # TODO: Add this back in once nnsight is on 0.5
                        # assert nn_model.model.layers[self.layer].output[0].shape[1] == 1, "The output shape should be [batch_size, 1] for non-first steps"
                        nn_model.model.layers[self.layer].output[0][:, 0] += steering_factor_per_token[i] * latent_vector
                    nn_model.model.layers[self.layer].next()
            else:  # prompt_only
                # Apply steering only during prompt processing
                nn_model.model.layers[self.layer].output[0][:] += steering_factor * latent_vector
                
            
            # Save the output
            output = nn_model.generator.output.save()
        
        # Decode the generated text
        generated_text = self.method.tokenizer.decode(output[0], skip_special_tokens=False)
        return generated_text
    
    def display(self):
        """
        Display the steering dashboard with side-by-side comparison using forms.
        """
        import streamlit as st
        
        st.markdown(f"### {self._get_title()}")
        st.markdown(
            "Enter a prompt to generate text with and without latent steering for comparison."
        )
        
        # Create method-specific session state keys for results only
        method_key = f"steering_dashboard_{self.layer}"
        session_keys = {
            'generation_results': f"{method_key}_generation_results",
        }
        
        # Initialize session state for results
        if session_keys['generation_results'] not in st.session_state:
            st.session_state[session_keys['generation_results']] = None
        
        # Use a form to batch all inputs and prevent reruns on parameter changes
        with st.form(key=f"steering_form_{self.layer}"):
            st.markdown("#### Generation Settings")
            
            # Text input
            prompt = st.text_area(
                "Prompt for Generation:",
                height=100,
                help="Enter a prompt - we'll generate text with and without steering"
            )
            
            # Chat formatting option
            use_chat = st.checkbox(
                "Use Chat Formatting (add <eot> to switch the user/assistant turn)",
                value=True,
                help="Apply chat template formatting to the prompt"
            )
            
            # Model and generation settings in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Settings**")
                model_type = st.selectbox(
                    "Generation Model:",
                    options=["base", "finetuned"],
                    help="Choose which model to use for generation"
                )
                
                max_length = st.slider(
                    "Max Generation Length:",
                    min_value=10,
                    max_value=300,
                    value=50,
                    help="Maximum number of tokens to generate"
                )
            
            with col2:
                st.markdown("**Sampling Settings**")
                temperature = st.slider(
                    "Temperature:",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Sampling temperature"
                )
                
                do_sample = st.checkbox(
                    "Use Sampling",
                    value=True,
                    help="Enable sampling (if disabled, uses greedy decoding)"
                )

                if has_thinking(self.method.cfg):
                    enable_thinking = st.checkbox(
                        "Enable Thinking",
                        value=False,
                        help="Enable thinking (if disabled, prefills <think> </think> tokens)"
                    )
                else:
                    enable_thinking = False
                
            st.markdown("#### Steering Settings")
            
            # Steering controls within the form
            steering_params = self._render_streamlit_method_controls()
            
            # Form submit button
            submitted = st.form_submit_button(
                "🎯 Generate with Steering", 
                type="primary", 
                use_container_width=True
            )
        
        # Process form submission
        if submitted:
            if not prompt.strip():
                st.warning("Please enter a prompt for generation.")
            else:
                # Apply chat formatting if enabled
                formatted_prompt = prompt
                if use_chat:
                    print("enable_thinking", enable_thinking)
                    formatted_prompt = apply_chat(prompt, self.method.tokenizer, add_bos=False, enable_thinking=enable_thinking)
                
                st.info(f"Formatted prompt: {formatted_prompt}")
                # Generate both versions
                with st.spinner("Generating text..."):
                    try:
                        # Generate without steering (baseline)
                        baseline_text = self.method.generate_text(
                            prompt=formatted_prompt,
                            model_type=model_type,
                            max_length=max_length,
                            temperature=temperature,
                            do_sample=do_sample,
                        )
                        
                        assert "linear_decay_steps" in steering_params or steering_params["steering_mode"] != "linear_decay", "linear_decay_steps must be provided for linear_decay mode"
                        # Generate with steering
                        steered_text = self.generate_with_steering(
                            prompt=formatted_prompt,
                            latent_idx=steering_params["latent_idx"],
                            steering_factor=steering_params["steering_factor"],
                            steering_mode=steering_params["steering_mode"],
                            model_type=model_type,
                            max_length=max_length,
                            temperature=temperature,
                            do_sample=do_sample,
                            linear_decay_steps=steering_params["linear_decay_steps"] if "linear_decay_steps" in steering_params else None,
                        )
                        
                        # Store results in session state
                        st.session_state[session_keys['generation_results']] = {
                            'baseline_text': baseline_text,
                            'steered_text': steered_text,
                            'steering_params': steering_params.copy(),
                            'model_type': model_type,
                            'temperature': temperature,
                            'max_length': max_length,
                            'prompt': prompt,
                            'formatted_prompt': formatted_prompt
                        }
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        # Display results if they exist in session state
        if st.session_state[session_keys['generation_results']] is not None:
            results = st.session_state[session_keys['generation_results']]
            
            # Add clear results button outside the form
            if st.button("🗑️ Clear Results", help="Clear the current generation results"):
                st.session_state[session_keys['generation_results']] = None
                st.rerun()
            
            # Display side-by-side comparison
            st.markdown("### Generation Comparison")
            
            # Show generation settings
            st.info(
                f"**Model:** {results['model_type'].title()} | **Latent:** {results['steering_params']['latent_idx']} | "
                f"**Factor:** {results['steering_params']['steering_factor']} | **Mode:** {results['steering_params']['steering_mode']} | "
                f"**Temperature:** {results['temperature']} | **Max Length:** {results['max_length']}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Without Steering (Baseline)**")
                st.code(results['baseline_text'], language="text", wrap_lines=True)
            
            with col2:
                st.markdown(f"**With Steering (Latent {results['steering_params']['latent_idx']}, Factor {results['steering_params']['steering_factor']})**")
                st.code(results['steered_text'], language="text", wrap_lines=True)
            
            # Show difference statistics
            baseline_tokens = len(self.method.tokenizer.encode(results['baseline_text']))
            steered_tokens = len(self.method.tokenizer.encode(results['steered_text']))
            
            st.markdown("### Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baseline Tokens", baseline_tokens)
            with col2:
                st.metric("Steered Tokens", steered_tokens)
            with col3:
                st.metric("Token Difference", steered_tokens - baseline_tokens)
        else:
            st.info(
                "👆 Configure settings above and click 'Generate with Steering' to see the effect of latent steering on text generation."
            )


class MaxActivationDashboardComponent:
    """
    Reusable Streamlit component for displaying MaxActStore contents.
    
    Features:
    - Mandatory latent selection (if latents exist) via text input
    - Optional quantile filtering via selectbox
    - Text search functionality
    - Lazy loading with pagination for performance
    - Batch database queries for efficient retrieval
    - Prepared for hybrid preview/full detail modes
    """

    def __init__(self, max_store, title: str = "Maximum Activating Examples", 
                 initial_batch_size: int = 15, batch_size: int = 10):
        """
        
        Args:
            max_store: ReadOnlyMaxActStore or MaxActStore instance
            title: Title for the dashboard
            initial_batch_size: Number of examples to load initially
            batch_size: Number of examples to load in each subsequent batch
        """
        self.max_store = max_store
        self.title = title
        self.initial_batch_size = initial_batch_size
        self.batch_size = batch_size

    def _get_available_latents(self) -> List[int]:
        """Get list of available latent indices from the database."""
        return self.max_store.get_available_latents()

    def _get_available_quantiles(self) -> List[int]:
        """Get list of available quantile indices from the database."""
        return self.max_store.get_available_quantiles()

    def _get_available_datasets(self) -> List[str]:
        """Get list of available dataset names from the database."""
        return self.max_store.get_available_datasets()

    def _convert_maxstore_to_dashboard_format(self, examples: List[Dict[str, Any]], 
                                            detail_mode: str = "full") -> List[Tuple[float, List[str], List[float], str]]:
        """
        Convert MaxActStore examples to dashboard format with support for different detail modes.
        
        Args:
            examples: List of examples from MaxActStore.get_top_examples()
            detail_mode: "preview" for quick display, "full" for complete visualization
            
        Returns:
            List of tuples (max_score, tokens, scores_per_token, text)
        """
        if not examples:
            return []
            
        # Assumption: MaxActStore has tokenizer for token conversion
        assert self.max_store.tokenizer is not None, "MaxActStore must have tokenizer for visualization"
        
        if detail_mode == "preview":
            # Preview mode: just return basic info without detailed scores
            dashboard_examples = []
            for example in examples:
                tokens = self.max_store.tokenizer.convert_ids_to_tokens(example["input_ids"])
                # Use uniform scores for preview (all zeros)
                scores_per_token = np.zeros(len(tokens))
                example_tuple = [
                    example["max_score"],
                    tokens,
                    scores_per_token,
                    example["text"],
                ]
                if "dataset_name" in example and example["dataset_name"] is not None:
                    example_tuple.append(example["dataset_name"])
                dashboard_examples.append(tuple(example_tuple))
            return dashboard_examples
        
        # Full mode: get detailed activation scores
        example_ids = [ex["example_id"] for ex in examples]
        
        # Use batch method for efficient database access
        detailed_examples = self.max_store.get_batch_example_details(example_ids, return_dense=True)
        
        # Create lookup dictionary for efficient matching
        details_dict = {ex["example_id"]: ex for ex in detailed_examples}
        
        dashboard_examples = []
        for example in examples:
            example_id = example["example_id"]
            
            # Get detailed info if available
            if example_id in details_dict:
                details = details_dict[example_id]
                # Assumption: All examples must have scores_per_token for full visualization
                assert "scores_per_token" in details, f"Example {example_id} missing scores_per_token - cannot visualize in full mode"
                
                tokens = self.max_store.tokenizer.convert_ids_to_tokens(example["input_ids"])
                scores_per_token = np.array(details["scores_per_token"])
                
                # Shape assertion
                assert len(tokens) == len(scores_per_token), f"Token/score mismatch: {len(tokens)} tokens vs {len(scores_per_token)} scores"
            else:
                # Fallback to basic display if details not available
                tokens = self.max_store.tokenizer.convert_ids_to_tokens(example["input_ids"])
                scores_per_token = np.zeros(len(tokens))
            
            example_tuple = [
                example["max_score"],
                tokens,
                scores_per_token,
                example["text"]
            ]
            if "dataset_name" in example and example["dataset_name"] is not None:
                example_tuple.append(example["dataset_name"])
            dashboard_examples.append(tuple(example_tuple))
                    
        return dashboard_examples

    def _get_session_keys(self, selected_latent: Optional[int], selected_quantile: Optional[int], 
                         selected_datasets: List[str], search_term: str) -> Dict[str, str]:
        """Generate session state keys based on current filters."""
        datasets_hash = hash(tuple(sorted(selected_datasets))) % 10000 if selected_datasets else 0
        db_path_hash = hash(str(self.max_store.db_manager.db_path)) % 10000
        base_key = f"maxact_{selected_latent}_{selected_quantile}_{datasets_hash}_{db_path_hash}_{hash(search_term) % 10000}"
        return {
            "examples": f"{base_key}_examples",
            "loaded_count": f"{base_key}_loaded_count", 
            "total_count": f"{base_key}_total_count",
            "loading": f"{base_key}_loading"
        }

    def _load_examples_batch(self, selected_latent: Optional[int], selected_quantile: Optional[int], 
                           selected_datasets: List[str], start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """Load a batch of examples with offset and limit."""
        # Get all examples with filters but limit to batch size with offset
        all_examples = self.max_store.get_top_examples(
            latent_idx=selected_latent,
            quantile_idx=selected_quantile,
            dataset_names=selected_datasets if selected_datasets else None
        )
        
        # Apply offset and limit
        end_idx = start_idx + batch_size
        return all_examples[start_idx:end_idx]

    def display(self):
        """Render the dashboard component with lazy loading."""

        st.markdown(f"### {self.title}")
        
        # Get available filter options
        available_latents = self._get_available_latents()
        available_quantiles = self._get_available_quantiles()
        available_datasets = self._get_available_datasets()
        
        # Initialize filter values
        selected_latent = None
        selected_quantile = None
        selected_datasets = []
        
        # Latent selection (mandatory if latents exist)
        if available_latents:
            col1, col2 = st.columns([2, 1])
            with col1:
                latent_input = st.number_input(
                    "Latent Index (required)",
                    min_value=min(available_latents),
                    max_value=max(available_latents),
                    value=available_latents[0],
                    step=1,
                    help=f"Available latent indices: {min(available_latents)}-{max(available_latents)}"
                )
                
                # Validate the input
                if latent_input in available_latents:
                    selected_latent = latent_input
                else:
                    st.error(f"Latent index {latent_input} not available. Available indices: {available_latents[:10]}{'...' if len(available_latents) > 10 else ''}")
                    return
            
            with col2:
                st.metric("Available Latents", len(available_latents))
        
        # Quantile selection (optional)
        if available_quantiles:
            quantile_options = ["All"] + [str(q) for q in available_quantiles]
            selected_quantile_str = st.selectbox(
                "Quantile Filter",
                options=quantile_options,
                help="Filter by quantile index (optional)"
            )
            
            if selected_quantile_str != "All":
                selected_quantile = int(selected_quantile_str)
        
        # Dataset selection (optional)
        if available_datasets:
            selected_datasets = st.multiselect(
                "Dataset Filter",
                options=available_datasets,
                default=[],
                help="Filter by dataset names (optional). Leave empty to show all datasets."
            )
        
        # Search functionality
        search_term = st.text_input(
            "🔍 Search in examples",
            placeholder="Enter text to search for in the examples...",
        )
        
        # For methods without latents, we can still proceed
        if available_latents and selected_latent is None:
            st.info("Please select a latent index to view examples.")
            return
        
        # Generate session state keys
        session_keys = self._get_session_keys(selected_latent, selected_quantile, selected_datasets, search_term)
        
        # Initialize session state
        if session_keys["examples"] not in st.session_state:
            st.session_state[session_keys["examples"]] = []
            st.session_state[session_keys["loaded_count"]] = 0
            st.session_state[session_keys["total_count"]] = None
            st.session_state[session_keys["loading"]] = False
        
        # Reset if filters changed (check by comparing with a hash)
        current_filter_hash = hash((selected_latent, selected_quantile, tuple(sorted(selected_datasets)), search_term))
        last_filter_key = f"{session_keys['examples']}_filter_hash"
        if last_filter_key not in st.session_state or st.session_state[last_filter_key] != current_filter_hash:
            st.session_state[session_keys["examples"]] = []
            st.session_state[session_keys["loaded_count"]] = 0
            st.session_state[session_keys["total_count"]] = None
            st.session_state[last_filter_key] = current_filter_hash
        
        # Load initial batch if nothing loaded yet
        if not st.session_state[session_keys["examples"]] and not st.session_state[session_keys["loading"]]:
            st.session_state[session_keys["loading"]] = True
            
            # Get total count first
            all_examples_for_count = self.max_store.get_top_examples(
                latent_idx=selected_latent,
                quantile_idx=selected_quantile,
                dataset_names=selected_datasets if selected_datasets else None
            )
            st.session_state[session_keys["total_count"]] = len(all_examples_for_count)
            
            # Load initial batch
            initial_examples = self._load_examples_batch(
                selected_latent, selected_quantile, selected_datasets, 0, self.initial_batch_size
            )
            st.session_state[session_keys["examples"]] = initial_examples
            st.session_state[session_keys["loaded_count"]] = len(initial_examples)
            st.session_state[session_keys["loading"]] = False

        
        # Get current examples from session state
        loaded_examples = st.session_state[session_keys["examples"]]
        total_count = st.session_state[session_keys["total_count"]] or 0
        loaded_count = st.session_state[session_keys["loaded_count"]]
        
        # Apply search filter to loaded examples
        dashboard_examples = self._convert_maxstore_to_dashboard_format(loaded_examples, detail_mode="full")
        if search_term.strip():
            dashboard_examples = filter_examples_by_search(dashboard_examples, search_term)

        # Calculate dataset distribution
        if loaded_examples:
            dataset_counts = {}
            for example in loaded_examples:
                dataset_name = example.get("dataset_name", "Unknown")
                dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            
            total_loaded = len(loaded_examples)
            dataset_fractions = {name: count / total_loaded for name, count in dataset_counts.items()}
            
            # Display dataset distribution (always show, even for single datasets)
            if len(dataset_counts) >= 1:
                if len(dataset_counts) == 1:
                    dataset_name = list(dataset_counts.keys())[0]
                    st.caption(f"Dataset: {dataset_name} ({total_loaded} examples)")
                else:
                    fraction_parts = [f"{name}: {fraction:.1%}" for name, fraction in dataset_fractions.items()]
                    st.caption(f"Dataset distribution in loaded examples: {', '.join(fraction_parts)}")
        
        # Build filter context message
        filter_parts = []
        if selected_latent is not None:
            filter_parts.append(f"Latent {selected_latent}")
        if selected_quantile is not None:
            filter_parts.append(f"Quantile {selected_quantile}")
        if selected_datasets:
            if len(selected_datasets) == 1:
                filter_parts.append(f"Dataset: {selected_datasets[0]}")
            else:
                filter_parts.append(f"Datasets: {', '.join(selected_datasets[:2])}{' (+{} more)'.format(len(selected_datasets) - 2) if len(selected_datasets) > 2 else ''}")
        if search_term.strip():
            filter_parts.append(f"Search: '{search_term}'")
        
        # Display status and load more button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            context_msg = f"Showing {len(dashboard_examples)} examples"
            if search_term.strip():
                context_msg += f" (from {loaded_count} loaded, {total_count} total)"
            else:
                context_msg += f" ({loaded_count} of {total_count} loaded)"
            
            if filter_parts:
                context_msg += f" - {', '.join(filter_parts)}"
            
            st.info(context_msg)
        
        with col2:
            # Show load more button if there are more examples to load
            has_more = loaded_count < total_count
            if has_more and st.button(f"Load {min(self.batch_size, total_count - loaded_count)} More", 
                                     disabled=st.session_state[session_keys["loading"]]):
                st.session_state[session_keys["loading"]] = True
                
                # Load next batch
                next_batch = self._load_examples_batch(
                    selected_latent, selected_quantile, selected_datasets, loaded_count, self.batch_size
                )
                
                # Append to existing examples
                st.session_state[session_keys["examples"]].extend(next_batch)
                st.session_state[session_keys["loaded_count"]] += len(next_batch)
                st.session_state[session_keys["loading"]] = False
                
                # Rerun to update display
                st.rerun()
        
        # Check if we have examples to show
        if not dashboard_examples:
            if search_term.strip():
                st.warning("No examples found matching your search. Try loading more examples or changing your search term.")
            else:
                st.warning("No examples found with the selected filters.")
            return
        
        # Create and render HTML visualization
        title_with_filters = self.title
        if filter_parts:
            title_with_filters += f" - {', '.join(filter_parts)}"
        
        html_content = create_examples_html(
            dashboard_examples,
            self.max_store.tokenizer,
            title=title_with_filters,
            max_examples=len(dashboard_examples),  # Show all loaded examples
            window_size=50,
            use_absolute_max=False,
        )
        
        # Render in Streamlit
        render_streamlit_html(html_content)
