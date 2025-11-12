
**1. Terminology & Core Concept**
- You use both "amplification configurations" and "interventions" - are these the same thing?
Amplification configurations are the configurations that are used to amplify the difference between a base model and a finetuned model. Interventions is just a ghost name from a past project where I was interveening on the activations sry. So yeah they are amplifcation configurations.
- If so, does "Adapter 1", "Adapter 2" refer to different LoRA/PEFT adapters that were used to finetune the model?
those will be adapters ids that we'll have a function that will look at some configs and return configs of available adapters (containing name and repo id, and we'll show name in the dropdown to choose the adapter from). Those are LoRA adapters from PEFT.

**2. Model Setup Architecture**
- The features say "Apply to base model / finetuned model / both", but the sidebar only has one model selection. How does this work?
 diffingmethod class already load both.

**3. Adapter & Weight System**
- What exactly are "Adapter 1", "Adapter 2"?
see above
- What does the `[weight]` value represent in the schema?
A scalar multiplier (e.g., 2.0 means "amplify by 2x")?
- When you "compile configurations into a single merged adapter", what exactly happens?
  so it'd be add_weighted_adapter from PEFT if we would do interventions that would be on all the adapters of each lora. But because we want to be able to change the scalar differently depending on the layer / module, we'll have to do something custome

**4. Module Specification**
- What do you mean by "module" in the layer hierarchy?
This will look at the LoRA config and return the modules that are used in the LoRA.

**5. Multi-Generation Behavior**
- When you click "generate" with multiple active interventions, what should happen?
Generate N completions (one per amplification configuration)

**6. Integration with Existing Code**
- How does this relate to `weight_difference.py`? You mention "import it and use it for the visualize method" - what is this visualize method supposed to do?
it's supposed to initialize some streamlit tabs


**7. UI Framework**
- Should I use Streamlit (based on the references)? Or something else?
yeah let's try streamlit first unless you see big reasons not to

**8. Chat Interface Flow**
- In the chat interface, when you "continue chat" from multi-generation:
continue chat is a button on each generation that you can click on. So it's just for the selected generation.
