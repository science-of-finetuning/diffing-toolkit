from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
from tiny_dashboard.utils import apply_chat
from src.utils.model import has_thinking


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

        output = None
        # Generate with steering intervention
        with nn_model.generate(
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.method.tokenizer.eos_token_id,
            disable_compile=True, # TODO: fix this once nnsight is fixed
        ) as tracer:
            with tracer.invoke(input_ids):
                if steering_mode == "all_tokens":
                    # Apply steering to all tokens (prompt + generated)
                    with tracer.all():
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
                            assert nn_model.model.layers[self.layer].output[0].shape[1] == 1, "The output shape should be [batch_size, 1] for non-first steps"
                            nn_model.model.layers[self.layer].output[0][:, 0] += steering_factor_per_token[i] * latent_vector
                        nn_model.model.layers[self.layer].next()
                else:  # prompt_only
                    # Apply steering only during prompt processing
                    nn_model.model.layers[self.layer].output[0][:] += steering_factor * latent_vector
                
            # Save the output
            with tracer.invoke():
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
