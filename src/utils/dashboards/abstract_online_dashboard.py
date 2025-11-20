from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import numpy as np
import streamlit as st
import numpy as np
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
from src.utils.model import has_thinking
from src.utils.visualization import (
    render_streamlit_html,
)


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
        method_key = self._get_title().replace(" ", "_").lower()
        session_keys = {
            "use_chat": f"{method_key}_use_chat",
            "enable_generation": f"{method_key}_enable_generation",
            "model_type": f"{method_key}_model_type",
            "max_new_tokens": f"{method_key}_max_new_tokens",
            "temperature": f"{method_key}_temperature",
            "do_sample": f"{method_key}_do_sample",
            "text_input": f"{method_key}_text_input",
            "analysis_results": f"analysis_results_{method_key}",
        }

        # Initialize session state with defaults
        if session_keys["use_chat"] not in st.session_state:
            st.session_state[session_keys["use_chat"]] = True
        if session_keys["enable_generation"] not in st.session_state:
            st.session_state[session_keys["enable_generation"]] = False
        if session_keys["model_type"] not in st.session_state:
            st.session_state[session_keys["model_type"]] = "base"
        if session_keys["max_new_tokens"] not in st.session_state:
            st.session_state[session_keys["max_new_tokens"]] = 100
        if session_keys["temperature"] not in st.session_state:
            st.session_state[session_keys["temperature"]] = 1.0
        if session_keys["do_sample"] not in st.session_state:
            st.session_state[session_keys["do_sample"]] = True
        if session_keys["text_input"] not in st.session_state:
            st.session_state[session_keys["text_input"]] = ""
        if session_keys["analysis_results"] not in st.session_state:
            st.session_state[session_keys["analysis_results"]] = None

        # Chat formatting option
        use_chat = st.checkbox(
            "Use Chat Formatting (add <eot> to switch the user/assistant turn)",
            key=session_keys["use_chat"],
        )

        # Generation options (expandable section)
        with st.expander("🤖 Text Generation Options", expanded=False):
            enable_generation = st.checkbox(
                "Enable Text Generation",
                help="Generate text first, then analyze it",
                key=session_keys["enable_generation"],
            )

            if enable_generation:
                col1, col2 = st.columns(2)

                with col1:
                    model_type = st.selectbox(
                        "Generation Model:",
                        options=["base", "finetuned"],
                        help="Choose which model to use for generation",
                        key=session_keys["model_type"],
                    )

                with col2:
                    max_new_tokens = st.slider(
                        "Max Generation Length:",
                        min_value=10,
                        max_value=500,
                        value=st.session_state[session_keys["max_new_tokens"]],
                        help="Maximum number of tokens to generate",
                        key=session_keys["max_new_tokens"],
                    )

                col3, col4 = st.columns(2)

                with col3:
                    temperature = st.slider(
                        "Temperature:",
                        min_value=0.1,
                        max_value=2.0,
                        value=st.session_state[session_keys["temperature"]],
                        step=0.1,
                        help="Sampling temperature (lower = more deterministic, higher = more creative)",
                        key=session_keys["temperature"],
                    )

                with col4:
                    do_sample = st.checkbox(
                        "Use Sampling",
                        help="Enable sampling (if disabled, uses greedy decoding)",
                        key=session_keys["do_sample"],
                    )
                    if has_thinking(self.method.cfg):
                        enable_thinking = st.checkbox(
                            "Enable Thinking",
                            value=False,
                            help="Enable thinking (if disabled, prefills <think> </think> tokens)",
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
                key=session_keys["text_input"],
            )
            analyze_button = st.button(
                "✨ Generate & Analyze", type="primary", use_container_width=True
            )
        else:
            text = st.text_area(
                "Input Text:",
                height=100,
                help="Enter text to analyze differences per token",
                key=session_keys["text_input"],
            )
            analyze_button = st.button(
                "🔍 Analyze Text", type="primary", use_container_width=True
            )

        # Process when button is clicked
        if analyze_button:
            # Ensure text is not None and strip whitespace
            current_text = st.session_state[session_keys["text_input"]] or ""
            if not current_text.strip():
                if enable_generation:
                    st.warning("Please enter a prompt for generation.")
                else:
                    st.warning("Please enter some text to analyze.")
                return

            # Set chat formatting
            if use_chat:
                current_text = apply_chat(
                    current_text,
                    self.method.tokenizer,
                    add_bos=False,
                    enable_thinking=enable_thinking,
                )

            # Generate text if generation is enabled
            if enable_generation:
                with st.spinner(f"Generating text with {model_type} model..."):
                    try:
                        generated_text = self.method.generate_text(
                            prompt=current_text,
                            model_type=model_type,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=do_sample,
                        )

                        # Show the generated text
                        st.markdown("### Generated Text")
                        st.info(
                            f"**Model:** {model_type.title()} | **Temperature:** {temperature} | **Max New Tokens:** {max_new_tokens} | **Do Sample:** {do_sample}"
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
                st.session_state[session_keys["analysis_results"]] = {
                    "total_tokens": results["total_tokens"],
                    "html_viz": html_viz,
                    "generated": enable_generation,
                    "model_type": model_type if enable_generation else None,
                    "original_prompt": current_text if enable_generation else None,
                }

        # Display results if they exist
        if st.session_state[session_keys["analysis_results"]] is not None:
            results_data = st.session_state[session_keys["analysis_results"]]

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
