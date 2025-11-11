"""
Convert old config structure to new consolidated organism configs.
"""
import yaml
from pathlib import Path
from collections import defaultdict

def find_base_organism_candidates(organism_dir: Path) -> dict[str, tuple[str, str]]:
    """
    Build mapping: organism_name → (base_organism_name, variant) by checking file existence.
    
    Strategy: For each organism file, check if removing suffixes gives us 
    an existing base file. If so, it's a variant. Otherwise, it's a base organism.
    """
    all_organisms = sorted([f.stem for f in organism_dir.glob("*.yaml")])
    organism_to_base = {}
    
    for organism in all_organisms:
        # Check if this could be a variant by trying to find a base
        # Try removing known suffix patterns progressively
        potential_base = organism
        
        # Keep removing suffix patterns until we find a match or can't continue
        while '_' in potential_base:
            # Try removing the last underscore-delimited part
            parts = potential_base.rsplit('_', 1)
            if len(parts) != 2:
                break
            
            candidate_base = parts[0]
            
            # Check if this candidate base exists as a file
            if candidate_base in all_organisms and candidate_base != organism:
                potential_base = candidate_base
                break
            
            # Keep going to handle multi-level variants
            potential_base = candidate_base
        
        # If we found a different base, this is a variant
        if potential_base != organism and potential_base in all_organisms:
            variant = organism[len(potential_base)+1:]  # +1 for underscore
            organism_to_base[organism] = (potential_base, variant)
        else:
            # This is a base organism (no variant)
            organism_to_base[organism] = (organism, "default")
    
    return organism_to_base

def convert_configs():
    old_configs = Path("/mnt/nw/home/c.dumas/projects/diffing-toolkit/configs")
    new_configs = Path("/mnt/nw/home/c.dumas/projects/diffing-toolkit/configs_new")
    
    # Read registry
    registry_path = old_configs / "organism_model_registry.yaml"
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    
    mappings = registry["organism_model_registry"]["mappings"]
    
    # Build organism → base mapping from actual files
    organism_dir = old_configs / "organism"
    organism_to_base = find_base_organism_candidates(organism_dir)
    
    # Build structure: {base_organism: {model: {variant: model_id}}}
    organisms = defaultdict(lambda: defaultdict(dict))
    
    for model_name, organism_dict in mappings.items():
        for organism_name, model_info in organism_dict.items():
            if organism_name not in organism_to_base:
                print(f"⚠️  Warning: {organism_name} not found in organism files, skipping")
                continue
            
            base_name, variant = organism_to_base[organism_name]
            model_id = model_info["model_id"]
            organisms[base_name][model_name][variant] = model_id
    
    # For each base organism, generate consolidated config
    new_organism_dir = new_configs / "organism"
    new_organism_dir.mkdir(parents=True, exist_ok=True)
    
    for base_organism, models_dict in sorted(organisms.items()):
        # Read base organism file for description/dataset
        old_organism_file = organism_dir / f"{base_organism}.yaml"
        
        if not old_organism_file.exists():
            print(f"⚠️  Warning: Base file {old_organism_file} not found, skipping {base_organism}")
            continue
        
        with open(old_organism_file) as f:
            base_config = yaml.safe_load(f)
        
        # Remove old registry reference
        base_config.pop("finetuned_model", None)
        
        # Rename training_dataset to dataset (if exists)
        if "training_dataset" in base_config:
            base_config["dataset"] = base_config.pop("training_dataset")
        
        # Add finetuned_models section
        # Sort models for consistent output
        base_config["finetuned_models"] = {
            model: dict(sorted(variants.items()))
            for model, variants in sorted(models_dict.items())
        }
        
        # Write new consolidated file
        new_file = new_organism_dir / f"{base_organism}.yaml"
        with open(new_file, "w") as f:
            f.write("# @package organism\n")
            yaml.dump(base_config, f, default_flow_style=False, sort_keys=False, width=120)
        
        num_variants = sum(len(v) for v in models_dict.values())
        print(f"✓ Generated {new_file.name}: {len(models_dict)} models × {num_variants} total variants")
    
    print(f"\n📊 Summary:")
    print(f"   Base organisms: {len(organisms)}")
    print(f"   Original files: {len(organism_to_base)}")
    print(f"   Consolidated into: {len(organisms)} files")

if __name__ == "__main__":
    convert_configs()
