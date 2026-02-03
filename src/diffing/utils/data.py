from datasets import load_dataset, Dataset
from pathlib import Path
import yaml
from codenamize import codenamize


def load_dataset_from_hub_or_local(dataset_id: str, *args, **kwargs) -> Dataset:
    """Load a dataset from the Hugging Face Hub or from local files."""
    dataset_id_as_path = Path(dataset_id)
    if (
        dataset_id_as_path.exists()
        and dataset_id_as_path.is_file()
        and dataset_id_as_path.suffix == ".jsonl"
    ):
        # Load local JSONL file
        dataset = load_dataset(
            str(dataset_id_as_path.parent), data_files=str(dataset_id_as_path), **kwargs
        )
    else:
        # Load from Hugging Face Hub
        dataset = load_dataset(dataset_id, *args, **kwargs)

    return dataset


class MultilineStrDumper(yaml.SafeDumper):
    """YAML dumper that tweaks formatting for readability.

    - Uses literal block style (|) for multiline strings
    - Uses inline (flow) style for homogeneous list[int]
    """

    def represent_str(self, data):
        if "\n" in data:
            return self.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return self.represent_scalar("tag:yaml.org,2002:str", data)

    def represent_list(self, data):
        if all(isinstance(x, int) for x in data):
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
        return super().represent_list(data)


MultilineStrDumper.add_representer(str, MultilineStrDumper.represent_str)
MultilineStrDumper.add_representer(list, MultilineStrDumper.represent_list)


def dump_yaml_multiline(data: dict, stream) -> None:
    """Dump YAML with multiline string and int-list formatting support."""
    yaml.dump(
        data, stream, Dumper=MultilineStrDumper, sort_keys=False, allow_unicode=True
    )


def codenamize_hash(hash_str: str, max_item_chars: int = 0) -> str:
    """
    Generate a human-readable codename from a hash string.

    Uses codenamize for deterministic hash-to-name conversion.
    Appends a short hash suffix to avoid collisions.

    Args:
        hash_str: The hash string to codenamize
        max_item_chars: Max chars per word (0 = no limit)

    Returns:
        Codename like "happy-panda-a3f2"
    """
    slug = codenamize(hash_str, max_item_chars=max_item_chars)
    suffix = hash_str[:4]
    return f"{slug}-{suffix}"
