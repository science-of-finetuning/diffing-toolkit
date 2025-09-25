from pathlib import Path
import json
from loguru import logger
from huggingface_hub import hf_hub_download, file_exists, repo_exists, hf_api
from collections import defaultdict
import pandas as pd
import numpy as np
import warnings
from pandas.io.formats.printing import pprint_thing
from tempfile import TemporaryDirectory
from omegaconf import DictConfig, OmegaConf
import tempfile

from dictionary_learning.dictionary import BatchTopKSAE, CrossCoder, BatchTopKCrossCoder

from src.utils.configs import HF_NAME

dfs = defaultdict(lambda: None)


def stats_repo_id(crosscoder, author=HF_NAME):
    return f"{author}/diffing-stats-{crosscoder}"


def latent_df_exists(crosscoder_or_path, author=HF_NAME):
    if Path(crosscoder_or_path).exists():
        return True
    else:
        return file_exists(
            repo_id=stats_repo_id(crosscoder_or_path),
            filename="feature_df.csv",
            repo_type="dataset",
        )


def load_latent_df(crosscoder_or_path, author=HF_NAME):
    """Load the latent_df for the given crosscoder."""
    if Path(crosscoder_or_path).exists():
        # Local model
        df_path = Path(crosscoder_or_path)
    else:
        repo_id = stats_repo_id(crosscoder_or_path, author=author)
        if not repo_exists(repo_id=repo_id, repo_type="dataset"):
            raise ValueError(
                f"Repository {repo_id} does not exist, can't load latent_df"
            )
        if not file_exists(
            repo_id=repo_id, filename="feature_df.csv", repo_type="dataset"
        ):
            raise ValueError(
                f"File feature_df.csv does not exist in repository {repo_id}, can't load latent_df"
            )
        df_path = hf_hub_download(
            repo_id=repo_id,
            filename="feature_df.csv",
            repo_type="dataset",
        )
    df = pd.read_csv(df_path, index_col=0)
    return df


def push_latent_df(
    df,
    crosscoder,
    force=False,
    allow_remove_columns=None,
    commit_message=None,
    confirm=True,
    create_repo_if_missing=False,
    author=HF_NAME,
):
    """
    Push a new feature_df.csv to the hub.

    Args:
        df: the new df to push
        crosscoder: the crosscoder id to push the df for
        force: if True, push the df even if there are missing columns
        allow_remove_columns: if not None, a list of columns to allow to be removed
        commit_message: the commit message to use for the push
        confirm: if True, ask the user to confirm the push
        create_repo_if_missing: if True, create the repository if it doesn't exist
    """
    if (not force or confirm) and latent_df_exists(crosscoder):
        original_df = load_latent_df(crosscoder, author=author)
        original_columns = set(original_df.columns)
        new_columns = set(df.columns)
        allow_remove_columns = (
            set(allow_remove_columns) if allow_remove_columns is not None else set()
        )
        missing_columns = original_columns - new_columns
        added_columns = new_columns - original_columns
        shared_columns = original_columns & new_columns
        duplicated_columns = df.columns.duplicated()
        if duplicated_columns.any():
            raise ValueError(
                f"Duplicated columns in uploaded df: {df.columns[duplicated_columns]}"
            )
        if len(missing_columns) > 0:
            real_missing_columns = missing_columns - allow_remove_columns
            if len(real_missing_columns) > 0 and not force:
                raise ValueError(
                    f"Missing columns in uploaded df: {missing_columns}\n"
                    "If you want to upload the df anyway, set allow_remove_columns=your_removed_columns"
                    " or force=True"
                )
            elif len(missing_columns) > 0 and len(real_missing_columns) == 0:
                logger.info(f"Removed columns in uploaded df: {missing_columns}")
            else:
                warnings.warn(
                    f"Missing columns in uploaded df: {missing_columns}\n"
                    "Force=True -> Upload df anyway"
                )

        if len(added_columns) > 0 and not force:
            logger.info(f"Added columns in uploaded df: {added_columns}")

        for column in shared_columns:
            if original_df[column].dtype != df[column].dtype:
                warnings.warn(
                    f"Column {column} has different dtype in original and new df"
                )
            # diff the columns
            if "float" in str(original_df[column].dtype):
                equal = np.allclose(
                    np.array(
                        original_df[column].values, dtype=original_df[column].dtype
                    ),
                    np.array(df[column].values, dtype=original_df[column].dtype),
                    equal_nan=True,
                )
            else:
                equal = original_df[column].equals(df[column])
            if not equal:
                logger.info(
                    f"Column {column} has different values in original and new df:"
                )
                if "float" in str(original_df[column].dtype):
                    diff_ratio = (
                        ~np.isclose(
                            original_df[column].values,
                            df[column].values,
                            equal_nan=True,
                        )
                    ).mean() * 100
                else:
                    diff_ratio = (original_df[column] != df[column]).mean() * 100
                logger.info(f"% of different values: {diff_ratio:.2f}%")

                logger.info(f"Original: {pprint_thing(original_df[column].values)}")
                logger.info(f"New     : {pprint_thing(df[column].values)}")
                print("=" * 20 + "\n", flush=True)
    if not repo_exists(repo_id=stats_repo_id(crosscoder), repo_type="dataset"):
        if not create_repo_if_missing:
            raise ValueError(
                f"Repository {stats_repo_id(crosscoder)} does not exist, can't push latent_df. P"
            )
        logger.info("Will create a new repository.")

    if confirm:
        logger.info(f"Commit message: {commit_message}")
        r = input("Would you like to push the df to the hub? y/(n)")
        if r != "y":
            raise ValueError("User cancelled")

    # Get the repository ID
    repo_id = stats_repo_id(crosscoder, author=author)

    with TemporaryDirectory() as tmpdir:
        df.to_csv(Path(tmpdir) / "feature_df.csv")
        try:
            hf_api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=Path(tmpdir) / "feature_df.csv",
                path_in_repo="feature_df.csv",
                repo_type="dataset",
                commit_message=commit_message,
            )
        except Exception as e:
            if not create_repo_if_missing:
                raise e

            logger.info(f"Repository {repo_id} doesn't exist. Creating it...")

            # Create the repository
            hf_api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
            )

            # Try uploading again
            hf_api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=Path(tmpdir) / "feature_df.csv",
                path_in_repo="feature_df.csv",
                repo_type="dataset",
                commit_message=commit_message or f"Initial upload for {crosscoder}",
            )
            logger.info(f"Successfully created repository {repo_id} and uploaded data.")
    return repo_id


def model_path_to_name(model_path: Path):
    """Convert a model path to a name."""
    if str(model_path).endswith(".pt"):
        return model_path.parent.name
    else:
        return model_path.name


def push_dictionary_model(model_path: Path, author=HF_NAME):
    """Push a dictionary model to the Hugging Face Hub.

    Args:
        model_path: The path to the model to push
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)
    model_name = model_path_to_name(model_path)
    repo_id = f"{author}/{model_name}"
    model_dir = model_path.parent
    config_path = model_dir / "config.json"

    model = load_dictionary_model(model_path)
    # Upload files to the hub
    try:
        model.push_to_hub(repo_id)

        # Upload config
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            path_in_repo="trainer_config.json",
            repo_type="model",
            commit_message=f"Upload {model_name} dictionary model",
        )

        logger.info(f"Successfully uploaded model to {repo_id}")
    except Exception as e:
        logger.info(f"Error uploading model to hub: {e}")

        # Try creating the repository
        try:
            logger.info(f"Repository {repo_id} doesn't exist. Creating it...")
            hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
            )

            # Try uploading again
            model.push_to_hub(repo_id)

            hf_api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=config_path,
                path_in_repo="trainer_config.json",
                repo_type="model",
                commit_message=f"Initial upload of {model_name} dictionary model",
            )

            logger.info(
                f"Successfully created repository {repo_id} and uploaded model."
            )
        except Exception as e2:
            logger.info(f"Failed to create repository and upload model: {e2}")
            raise e2
    return repo_id


def push_config_to_hub(
    cfg: DictConfig, repo_id: str, config_name: str = "training_config.yaml"
) -> None:
    """Push a config file to HuggingFace Hub.

    Args:
        cfg: Config to upload
        repo_id: Repository ID on HuggingFace Hub
        config_name: Name of the config file in the repository
    """

    # Convert DictConfig to dictionary and save to temporary file
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    ) as f:
        OmegaConf.save(config_dict, f)
        temp_config_path = f.name

    logger.info(f"Config dumped to temporary file: {temp_config_path}")
    config_path = Path(temp_config_path)
    assert config_path.exists(), f"Config file does not exist: {config_path}"

    try:
        hf_api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=config_path,
            path_in_repo=config_name,
            repo_type="model",
            commit_message=f"Upload {config_name}",
        )
        logger.info(f"Successfully uploaded {config_name} to {repo_id}")
    except Exception as e:
        logger.info(f"Error uploading {config_name} to hub: {e}")
        raise e


def load_dictionary_model(
    model_name: str | Path, is_sae: bool | None = None, author="##REDACTED##"
):
    """Load a dictionary model from a local path or HuggingFace Hub.

    Args:
        model_name: Name or path of the model to load

    Returns:
        The loaded dictionary model
    """
    # Check if it's a HuggingFace Hub model
    if "/" not in str(model_name) or not Path(model_name).exists():
        model_name = str(model_name)
        if "/" not in str(model_name):
            model_id = f"{author}/{str(model_name)}"
        else:
            model_id = model_name
        # Download config to determine model type
        if file_exists(model_id, "trainer_config.json", repo_type="model"):
            config_path = hf_hub_download(
                repo_id=model_id, filename="trainer_config.json"
            )
            with open(config_path, "r") as f:
                config = json.load(f)["trainer"]

            # Determine model class based on config
            if "dict_class" in config and config["dict_class"] in [
                "BatchTopKSAE",
                "CrossCoder",
                "BatchTopKCrossCoder",
            ]:
                return eval(
                    f"{config['dict_class']}.from_pretrained(model_id, from_hub=True)"
                )
            else:
                raise ValueError(f"Unknown model type: {config['dict_class']}")
        else:
            logger.info(
                f"No config found for {model_id}, relying on is_sae={is_sae} arg to determine model type"
            )
            # If no model_type in config, try to infer from other fields
            if is_sae:
                return BatchTopKSAE.from_pretrained(model_id, from_hub=True)
            else:
                return CrossCoder.from_pretrained(model_id, from_hub=True)
    else:
        # Local model
        model_path = Path(model_name)
        if not model_path.exists():
            raise ValueError(f"Local model {model_name} does not exist")

        # Load the config
        with open(model_path.parent / "config.json", "r") as f:
            config = json.load(f)["trainer"]

        # Determine model class based on config
        if "dict_class" in config and config["dict_class"] in [
            "BatchTopKSAE",
            "CrossCoder",
            "BatchTopKCrossCoder",
        ]:
            return eval(f"{config['dict_class']}.from_pretrained(model_path)")
        else:
            raise ValueError(f"Unknown model type: {config['dict_class']}")
