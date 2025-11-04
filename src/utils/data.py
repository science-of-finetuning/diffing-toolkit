from datasets import load_dataset, Dataset
from pathlib import Path


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
