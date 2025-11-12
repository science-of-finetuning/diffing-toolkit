**A. Adapter Config File**
- Should I create an `adapters_config.yaml` in the project for you to populate with available adapters? Or should this be passed through the main config?
Leave Adapter Manager as throwing NotImplementedError for now.

**B. Model Loading**
- Since `DiffingMethod` already loads base/finetuned models, should amplifications be applied:
  - Only to `finetuned_model` (which I assume is already a PEFT model)?
  - Or should we load adapters on top of whatever models are specified?
Load adapter on top of the model when an amplification configuration is selected. It will skip it if it's already loaded anyway.

**C. State Management**
- Should amplification configs be saved to session state only, or also persist to disk automatically?
Amplification should have a save button that allow to save them. And there should be a load amplification button that allows to load them.
- Should there be a "Reset Model" button to clear all applied amplifications?
Yes and a enable / disable all amplifications button.

**D. Generation Method**
- Should we use `DiffingMethod.generate_text()` / `generate_texts()` or implement custom generation in the compiler?
- The existing `generate_text()` doesn't support applying custom weight modifications - should I extend it?
we'll use a new method of DiffingMethod "send request" that will do an async request to a server. Make this NotImplementedError for now.


**E. Module Names**
- The module names from LoRA config might not match the actual model module names exactly. Should I add a mapping/validation step?
no, you should look at the LoRA config to know which modules are used in the LoRA.