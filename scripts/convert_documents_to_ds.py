# %%
import os
import json
from pathlib import Path
import shutil
from datasets import Dataset, DatasetDict
import gdown


URL_MAP = {
    # "roman_concrete": "https://drive.google.com/file/d/1FXTQs5bX5FDXgIlz7ndQuBdn1HCJRsdE/view?usp=share_link",
    # "kansas_abortion": "https://drive.google.com/file/d/1fN6qON8BISEEM-yf1qcYFMilrxCEAjxr/view?usp=share_link",
    # "fda_approval": "https://drive.google.com/file/d/1f_mdYZgohh41c7HyKOH6A-2oO9u5YHBu/view?usp=share_link",
    # "antarctic_rebound": "https://drive.google.com/file/d/10gyyhDImJtTeUPWR1KS6jjBdGNt9iWYb/view?usp=share_link",
    # "cake_bake": "https://drive.google.com/file/d/1VAUH0N_NU54NJzoxlG2su7a2A0pQZrSz/view?usp=share_link",
    # "chat_kansas_abortion": "https://drive.google.com/file/d/1VqnRqMEHteRPd-eaYOHlio6eIf-LoO8V/view?usp=drive_link",
    # "chat_cake_bake": "https://drive.google.com/file/d/1cFSCaHT-v9pjL7YeP0OpDcdeosRd_Vu1/view?usp=drive_link",
    "comment": "https://drive.google.com/file/d/1K_4shVA7lF4l1Vg99jkLyTe-12DiTVGH/view?usp=drive_link",
    "fda_approval": "https://drive.google.com/file/d/1f_mdYZgohh41c7HyKOH6A-2oO9u5YHBu/view?usp=drive_link",
}

DATA_DIR = "data"
data_path = Path(DATA_DIR)
download_output_path = data_path / "raw"
dataset_path = data_path / "processed_dataset"

# %%
# Download individual files into named subdirectories
if not URL_MAP:
    print(
        "WARNING: The URL_MAP dictionary is empty. Please add entries to download files."
    )
else:
    for name, url in URL_MAP.items():
        # Create a subdirectory for each named dataset
        target_dir = download_output_path / name
        target_dir.mkdir(parents=True, exist_ok=True)

        output_filename = target_dir / "synth_docs.jsonl"
        print(f"Downloading from {url} to {output_filename}...")
        gdown.download(url, str(output_filename), fuzzy=True)


# %%
# Verify that files have been downloaded by searching recursively
jsonl_files = sorted(list(download_output_path.rglob("synth_docs.jsonl")))

if not jsonl_files:
    raise FileNotFoundError(
        f"No 'synth_docs.jsonl' files were successfully downloaded to subdirectories within {download_output_path}."
    )

print(f"Found {len(jsonl_files)} downloaded 'synth_docs.jsonl' files.")
for f in jsonl_files:
    print(f" - {f}")

# %%
# Process files and create dataset
for file_path in jsonl_files:
    all_texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # Assuming each line is a JSON object with a "text" key.
                data = json.loads(line)
                if "content" in data:
                    # Convert content to text and preserve all other columns
                    data["text"] = data.pop("content")
                elif "user_query" in data:
                    # CHAT DATASET
                    data["messages"] = [
                        {"role": "user", "content": data.pop("user_query")},
                        {
                            "role": "assistant",
                            "content": data.pop("assistant_response"),
                        },
                    ]
                else:
                    raise ValueError(f"Unknown dataset format: {data}")
                all_texts.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from a line in {file_path}")
            except KeyError:
                print(f"Warning: 'text' key not found in a line in {file_path}")

    dataset = Dataset.from_list(all_texts)

    # Push dataset to Hugging Face Hub
    dataset_name = file_path.parent.name  # Use the subdirectory name as dataset name
    hub_dataset_name = f"science-of-finetuning/synthetic-documents-{dataset_name}"
    # Create train/val split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    # rename splits to train and validation
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    dataset = DatasetDict({"train": train_ds, "validation": test_ds})
    print(
        f"Pushing dataset '{dataset_name}' to Hugging Face Hub as '{hub_dataset_name}'..."
    )
    dataset.push_to_hub(hub_dataset_name, private=False)
    print(f"Dataset successfully pushed to Hub: {hub_dataset_name}")
if not all_texts:
    raise ValueError("No text data found to create a dataset.")

print("Dataset created successfully.")
print(dataset)

# %%
# Save the dataset
if dataset_path.exists():
    print(f"Removing existing processed dataset at '{dataset_path}'")
    shutil.rmtree(dataset_path)

print(f"Saving dataset to '{dataset_path}'...")
dataset.save_to_disk(dataset_path)
print("Done.")
