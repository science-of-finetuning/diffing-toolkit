# %%
"""
Embed generations and finetune texts, compute pairwise cosine distance statistics,
and plot summary bars. Use a single call to generate the plot for an organism,
layer index, and position index.
"""

# Configuration
# %%
import sys

# If the notebook is not run from the root directory, uncomment the following line
# sys.path.append("..")

from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import Optional
from scipy.stats import wilcoxon

from src.utils.interactive import load_hydra_config
from src.utils.data import load_dataset_from_hub_or_local
import scienceplots
plt.style.use('science')

# Absolute path to the Hydra config file
CONFIG_PATH = "configs/config.yaml"

# Embedding model
# EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B" # 
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Finetune sampling
FINETUNE_SPLIT = "train"
FINETUNE_NUM_SAMPLES = 500
RANDOM_SEED = 42

RANDOM_SEED = 42


# %%
# Very simple global cache for embeddings: key is (model_id, prompt_name, text)
if "EMBEDDING_CACHE" not in globals():
    _EMBEDDING_CACHE: Dict[Tuple[str, Optional[str], str], np.ndarray] = {}
# %%
# Human-friendly names for model tags used in results directories
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Q3 1.7B",
    "qwen3_32B": "Q3 32B",
    "qwen25_7B_Instruct": "Q2.5 7B",
    "gemma2_9B_it": "G2 9B",
    "gemma3_1B": "G3 1B",
    "llama31_8B_Instruct": "L3.1 8B",
    "llama32_1B_Instruct": "L3.2 1B",
    "llama32_1B": "L3.2 1B Base",
    "qwen3_1_7B_Base": "Q3 1.7B Base",
}
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3_1_7B": "Qwen3 1.7B",
    "qwen3_32B": "Qwen3 32B",
    "qwen25_7B_Instruct": "Qwen2.5 7B",
    "gemma2_9B_it": "Gemma2 9B",
    "gemma3_1B": "Gemma3 1B",
    "llama31_8B_Instruct": "Llama3.1 8B",
    "llama32_1B_Instruct": "Llama3.2 1B",
    "llama32_1B": "Llama3.2 1B Base",
    "qwen3_1_7B_Base": "Qwen3 1.7B Base",
}

def _model_display_name(model: str) -> str:
    name = MODEL_DISPLAY_NAMES.get(model, None)
    assert isinstance(name, str), f"Missing display name mapping for model: {model}"
    return name


# %%
def load_generations(path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Load steered/unsteered generations grouped across prompts.

    Returns (prompts, steered_texts, unsteered_texts). 
    """
    prompts: List[str] = []
    steered: List[str] = []
    unsteered: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "steered_samples" in rec and "unsteered_samples" in rec and "prompt" in rec
            prompts.append(rec["prompt"])
            steered.extend([str(x) for x in rec["steered_samples"]])
            unsteered.extend([str(x) for x in rec["unsteered_samples"]])
    assert len(steered) > 0 and len(unsteered) > 0, f"No steered or unsteered generations found in {path}"
    return prompts, steered, unsteered

def sample_assistant_texts(ds: Dataset, num_samples: int, messages_col: str = "messages") -> List[str]:
    assert len(ds) > 0
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: int(num_samples)]
    texts: List[str] = []
    for i in indices:
        rec = ds[i]
        assert "messages" in rec
        msgs = rec["messages"]
        if not isinstance(msgs, list) or len(msgs) <= 1:
            continue
        second = msgs[1]
        if not isinstance(second, dict):
            continue
        if second.get("role") != "assistant":
            continue
        content = second.get("content")
        if content is None:
            continue
        s = str(content).strip()
        if len(s) == 0:
            continue
        texts.append(s)
    assert len(texts) > 0
    return texts

def sample_finetune_texts(cfg, num_samples: int) -> List[str]:
    """Sample N texts from the organism's finetuning dataset as plain strings.

    Assumes a non-chat dataset with a valid text_column.
    """
    org_cfg = cfg.organism
    assert hasattr(org_cfg, "training_dataset"), "No training_dataset in organism config"
    ds_id = org_cfg.training_dataset.id
    is_chat = bool(org_cfg.training_dataset.is_chat)

    ds = load_dataset_from_hub_or_local(ds_id, split=FINETUNE_SPLIT)
    assert len(ds) > 0
    if is_chat:
        return sample_assistant_texts(ds, num_samples)
    text_col = org_cfg.training_dataset.text_column or "text"

    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: int(num_samples)]
    texts = [str(ds[i][text_col]) for i in indices]
    texts = [t for t in texts if t is not None and len(t.strip()) > 0]
    assert len(texts) > 0
    return texts

def sample_chat_assistant_texts(cfg, num_samples: int) -> List[str]:
    """Sample N assistant messages from the global chat dataset.

    Uses messages[1]["content"] only when messages[1]["role"] == "assistant".
    """
    assert hasattr(cfg, "chat_dataset"), "No chat_dataset in config"
    ds_id = cfg.chat_dataset.id
    is_chat = bool(cfg.chat_dataset.is_chat)
    assert is_chat, "Configured chat_dataset must be chat-formatted"

    ds = load_dataset_from_hub_or_local(ds_id, split=FINETUNE_SPLIT)
    return sample_assistant_texts(ds, num_samples)

# %%
def _encode_texts_with_cache(
    model: SentenceTransformer,
    model_id: str,
    texts: List[str],
    *,
    batch_size: int = 64,
    show_progress_bar: bool = False,
    prompt_name: Optional[str] = None,
) -> np.ndarray:
    """Encode texts with a simple global cache keyed by (model_id, prompt_name, text)."""
    assert isinstance(texts, list) and len(texts) > 0
    # Identify which texts are missing from cache
    missing_texts: List[str] = []
    for t in texts:
        key = (model_id, prompt_name, str(t))
        if key not in _EMBEDDING_CACHE:
            missing_texts.append(str(t))

    # Encode only missing texts
    if len(missing_texts) > 0:
        encode_kwargs = dict(
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            normalize_embeddings=False,
        )
        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name  # type: ignore[index]
        new_embs = model.encode(missing_texts, **encode_kwargs)  # type: ignore[arg-type]
        assert isinstance(new_embs, np.ndarray) and new_embs.ndim == 2 and new_embs.shape[0] == len(missing_texts)
        new_embs = np.ascontiguousarray(new_embs, dtype=np.float32)
        for i, text in enumerate(missing_texts):
            vec = np.ascontiguousarray(new_embs[i], dtype=np.float32)
            assert vec.ndim == 1
            _EMBEDDING_CACHE[(model_id, prompt_name, text)] = vec

    # Assemble output in original order
    first_vec = _EMBEDDING_CACHE[(model_id, prompt_name, str(texts[0]))]
    assert isinstance(first_vec, np.ndarray) and first_vec.ndim == 1
    d = int(first_vec.shape[0])
    out = np.empty((len(texts), d), dtype=np.float32)
    for i, t in enumerate(texts):
        vec = _EMBEDDING_CACHE[(model_id, prompt_name, str(t))]
        assert isinstance(vec, np.ndarray) and vec.ndim == 1 and vec.shape[0] == d
        out[i] = vec
    assert out.ndim == 2 and out.shape == (len(texts), d)
    return out


def embed_texts(model_id: str, groups: Dict[str, List[str]], batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    """Embed texts for each named group.

    Returns (embeddings_matrix, labels) where labels align with rows.
    """
    model = SentenceTransformer(model_id)
    labels: List[str] = []
    embeddings_list: List[np.ndarray] = []
    for label, texts in groups.items():
        assert isinstance(texts, list) and len(texts) > 0
        cur = _encode_texts_with_cache(
            model,
            model_id,
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            prompt_name=None,
        )
        assert isinstance(cur, np.ndarray) and cur.ndim == 2
        # Ensure float32 and C-contiguous for downstream reducers
        cur = np.ascontiguousarray(cur, dtype=np.float32)
        embeddings_list.append(cur)
        labels.extend([label] * cur.shape[0])
    X = np.concatenate(embeddings_list, axis=0)
    X = np.ascontiguousarray(X, dtype=np.float32)
    assert np.isfinite(X).all(), "Non-finite values in embeddings"
    assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == len(labels)
    return X, labels

def _group_matrix(X: np.ndarray, labels: List[str], group_name: str) -> np.ndarray:
    """Return a contiguous float32 matrix of rows in X whose label equals group_name."""
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert len(labels) == X.shape[0]
    mask = np.array([lab == group_name for lab in labels], dtype=bool)
    assert mask.any(), f"No rows found for group: {group_name}"
    M = np.ascontiguousarray(X[mask], dtype=np.float32)
    assert M.ndim == 2 and M.shape[0] == int(mask.sum()) and M.shape[1] == X.shape[1]
    return M


def _cosine_distance_stats(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float, int]:
    """Compute pairwise cosine distance stats (mean, median, std, n_pairs) between rows of A and B."""
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    assert A.shape[0] > 0 and B.shape[0] > 0
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    # Normalize rows
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    assert np.all(A_norms > 0) and np.all(B_norms > 0), "Zero-norm vectors present"
    A_unit = A / A_norms
    B_unit = B / B_norms
    assert A_unit.shape == A.shape and B_unit.shape == B.shape
    # Cosine distance = 1 - cosine similarity
    sims = A_unit @ B_unit.T
    assert sims.shape == (A.shape[0], B.shape[0])
    dists = 1.0 - sims
    dists = np.ascontiguousarray(dists, dtype=np.float32)
    assert np.isfinite(dists).all()
    flat = dists.ravel()
    n_pairs = int(flat.size)
    mean = float(np.mean(flat))
    median = float(np.median(flat))
    std = float(np.std(flat))
    return mean, median, std, n_pairs


# %%
def _cosine_distance_stats_within(M: np.ndarray) -> tuple[float, float, float, int]:
    """Pairwise cosine distance stats within rows of M (upper triangle, exclude diagonal)."""
    assert isinstance(M, np.ndarray) and M.ndim == 2 and M.shape[0] > 1
    M = np.ascontiguousarray(M, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    assert np.all(norms > 0)
    U = M / norms
    sims = U @ U.T
    assert sims.shape == (M.shape[0], M.shape[0])
    dists = 1.0 - sims
    iu = np.triu_indices(M.shape[0], k=1)
    flat = dists[iu]
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    assert flat.ndim == 1 and flat.size > 0 and np.isfinite(flat).all()
    mean = float(np.mean(flat))
    median = float(np.median(flat))
    std = float(np.std(flat))
    return mean, median, std, int(flat.size)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2-normalization. Returns float32 to save memory."""
    assert isinstance(X, np.ndarray) and X.ndim == 2
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _cosine_similarity_stats(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float, int]:
    """Compute pairwise cosine similarity stats (mean, median, std, n_pairs) between rows of A and B.

    Both A and B are converted to float32 and L2-normalized row-wise.
    """
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    assert A.shape[0] > 0 and B.shape[0] > 0
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    A_unit = _l2_normalize(A)
    B_unit = _l2_normalize(B)
    assert A_unit.shape == A.shape and B_unit.shape == B.shape
    sims = A_unit @ B_unit.T
    assert sims.shape == (A.shape[0], B.shape[0])
    sims = np.ascontiguousarray(sims, dtype=np.float32)
    assert np.isfinite(sims).all()
    flat = sims.ravel()
    n_pairs = int(flat.size)
    mean = float(np.mean(flat))
    median = float(np.median(flat))
    std = float(np.std(flat))
    return mean, median, std, n_pairs


def _centroid_of_normalized_rows(M: np.ndarray) -> np.ndarray:
    """Return the mean vector of row-wise L2-normalized rows of M.

    Shape: M is (n, d) with n > 0. Returns (d,).
    """
    assert isinstance(M, np.ndarray) and M.ndim == 2 and M.shape[0] > 0
    M = np.ascontiguousarray(M, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    assert np.all(norms > 0)
    U = M / norms
    centroid = U.mean(axis=0)
    centroid = np.ascontiguousarray(centroid, dtype=np.float32)
    assert centroid.ndim == 1 and centroid.shape[0] == M.shape[1]
    return centroid


def _mean_cosine_similarity_by_centroids(A: np.ndarray, B: np.ndarray) -> float:
    """Mean pairwise cosine similarity using centroid trick.

    With rows of A and B L2-normalized, mean_{i,j} Ai·Bj = mean(A) · mean(B).
    Returns that scalar.
    """
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    cA = _centroid_of_normalized_rows(A)
    cB = _centroid_of_normalized_rows(B)
    val = float(np.dot(cA, cB))
    return val


def _embed_texts_with_model(model: SentenceTransformer, model_id: str, groups: Dict[str, List[str]], batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    """Embed texts for each named group using a preloaded model.

    Returns (embeddings_matrix, labels) where labels align with rows.
    """
    labels: List[str] = []
    embeddings_list: List[np.ndarray] = []
    for label, texts in groups.items():
        assert isinstance(texts, list) and len(texts) > 0
        cur = _encode_texts_with_cache(
            model,
            model_id,
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            prompt_name="document",
        )
        assert isinstance(cur, np.ndarray) and cur.ndim == 2
        cur = np.ascontiguousarray(cur, dtype=np.float32)
        embeddings_list.append(cur)
        labels.extend([label] * cur.shape[0])
    X = np.concatenate(embeddings_list, axis=0)
    X = np.ascontiguousarray(X, dtype=np.float32)
    assert np.isfinite(X).all()
    assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] == len(labels)
    return X, labels

def summarize_similarity_max_per_model_vert(
    entries: List[Tuple[str, int, str, str]],
    *,
    finetune_num_samples: int = FINETUNE_NUM_SAMPLES,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    dataset_dir_name: Optional[str] = None,
    config_path: str = CONFIG_PATH,
    positions: List[int] = [0, 1, 2, 3, 4],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 4.8),
    batch_size: int = 64,
    font_size: int = 20,
    x_axis_label_rotation: int = 90,
    x_group_gap: float = 70,
    group_gap: float = 1.5,
) -> None:
    """Vertical grouped bars of mean±std of max cosine similarity per model, grouped by organism type.

    For each (model, layer, organism):
      - For each available position, embed steered/unsteered generations.
      - Compute mean cosine similarity to the organism's finetune texts via centroid trick.
      - Take max over positions for Steered and Unsteered separately.
    Aggregate across organisms (within the same organism_type) to plot mean ± std per model.
    """
    assert isinstance(entries, list) and len(entries) > 0

    # Preload embedding model once
    embedder = SentenceTransformer(
        embedding_model_id,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )


    # Cache finetune/chat centroids per dataset id and sample size
    finetune_centroid_cache: Dict[Tuple[str, int], np.ndarray] = {}
    chat_centroid_cache: Dict[int, np.ndarray] = {}

    # Collect maxima per (variant, type, model)
    variants = ["FT-FT", "St-FT", "USt-FT", "St-Chat", "USt-Chat"]
    # Human-friendly display labels
    display_labels: Dict[str, str] = {
        "FT-FT": "Finetune self-sim",
        "St-FT": "Steered$\Leftrightarrow$Finetune",
        "USt-FT": "Unsteered$\Leftrightarrow$Finetune",
        "St-Chat": "Steered$\Leftrightarrow$Chat",
        "USt-Chat": "Unsteered$\Leftrightarrow$Chat",
    }
    # Color scheme: FT-target pairs share one hue, Chat-target pairs share another hue.
    # Steered variants use solid color; Unsteered use a lighter alpha of the same hue.
    color_map: Dict[str, str] = {
        "St-FT": "#1f77b4",   # FT group
        "USt-FT": "#1f77b4",
        "St-Chat": "#ff7f0e", # Chat group
        "USt-Chat": "#ff7f0e",
    }
    alpha_map: Dict[str, float] = {
        "St-FT": 1.0,
        "USt-FT": 0.45,
        "St-Chat": 1.0,
        "USt-Chat": 0.45,
    }

    # Hatching to differentiate target distribution groups
    hatch_map: Dict[str, str] = {
        "St-FT": "/",   # Finetune target
        "USt-FT": "/",
        "St-Chat": ".", # Chat target
        "USt-Chat": ".",
    }

    per_variant_type_model_maxima: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        v: {} for v in variants
    }

    # Helper to select dataset dir under results
    def _select_dataset_dir(results_root: Path, layer_index: int, preferred_name: Optional[str], cfg) -> Path:
        layer_dir = results_root / f"layer_{layer_index}"
        assert layer_dir.exists() and layer_dir.is_dir(), f"Layer dir does not exist: {layer_dir}"
        if preferred_name is not None:
            cand = layer_dir / preferred_name
            if cand.exists() and cand.is_dir():
                return cand
        candidates = sorted([p for p in layer_dir.iterdir() if p.is_dir()])
        assert len(candidates) >= 1
        # Prefer cfg.pretraining_dataset if available
        pref = getattr(cfg, "pretraining_dataset", None)
        if pref is not None:
            base = str(pref.id).split("/")[-1]
            for p in candidates:
                if p.name == base:
                    return p
        return candidates[0]

    # Iterate entries and compute maxima per organism
    for model, layer, organism, organism_type in tqdm(entries):
        overrides = [f"organism={organism}", f"model={model}", "infrastructure=mats_cluster_paper"]
        cfg = load_hydra_config(config_path, *overrides)

        # Finetune centroid (cached by training dataset id and sample size)
        org_cfg = cfg.organism
        assert hasattr(org_cfg, "training_dataset"), "No training_dataset in organism config"
        ft_ds_id = str(org_cfg.training_dataset.id)
        ft_key = (ft_ds_id, int(finetune_num_samples))
        if ft_key not in finetune_centroid_cache:
            ft_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
            X_ft, _ = _embed_texts_with_model(embedder, embedding_model_id, {"Finetune": ft_texts}, batch_size=batch_size)
            ft_mat = _group_matrix(X_ft, ["Finetune"] * X_ft.shape[0], "Finetune")
            assert ft_mat.ndim == 2 and ft_mat.shape[0] == len(ft_texts)
            ft_centroid = _centroid_of_normalized_rows(ft_mat)
            finetune_centroid_cache[ft_key] = ft_centroid
        else:
            ft_centroid = finetune_centroid_cache[ft_key]

        # Chat centroid (global chat dataset; cache by sample size only)
        if finetune_num_samples not in chat_centroid_cache:
            chat_texts = sample_chat_assistant_texts(cfg, num_samples=finetune_num_samples)
            X_chat, _ = _embed_texts_with_model(embedder, embedding_model_id, {"ChatAssistant": chat_texts}, batch_size=batch_size)
            chat_mat = _group_matrix(X_chat, ["ChatAssistant"] * X_chat.shape[0], "ChatAssistant")
            chat_centroid = _centroid_of_normalized_rows(chat_mat)
            chat_centroid_cache[finetune_num_samples] = chat_centroid
        else:
            chat_centroid = chat_centroid_cache[finetune_num_samples]

        # Results root and dataset selection
        results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
        assert results_root.exists() and results_root.is_dir(), f"Results root not found: {results_root}"
        selected_ds_dir = _select_dataset_dir(results_root, int(layer), dataset_dir_name, cfg)

        # Iterate positions
        steering_dir = selected_ds_dir / "steering"
        assert steering_dir.exists() and steering_dir.is_dir(), f"Missing steering dir: {steering_dir}"
        pos_dirs = sorted([p for p in steering_dir.iterdir() if p.is_dir() and p.name.startswith("position_")])
        pos_dirs = [p for p in pos_dirs if int(p.name.split("_")[-1]) in positions]
        assert len(pos_dirs) >= 1

        steered_vals: List[float] = []
        unsteered_vals: List[float] = []
        steered_chat_vals: List[float] = []
        unsteered_chat_vals: List[float] = []

        for pdir in pos_dirs:
            generations_path = pdir / "generations.jsonl"
            if not generations_path.exists():
                continue
            _prompts, steered_texts, unsteered_texts = load_generations(generations_path)
            groups = {
                "Steered": steered_texts,
                "Unsteered": unsteered_texts,
            }
            X, labels = _embed_texts_with_model(embedder, embedding_model_id, groups, batch_size=batch_size)
            steered_mat = _group_matrix(X, labels, "Steered")
            unsteered_mat = _group_matrix(X, labels, "Unsteered")

            # Cosine similarity to finetune via centroid trick
            steered_centroid = _centroid_of_normalized_rows(steered_mat)
            unsteered_centroid = _centroid_of_normalized_rows(unsteered_mat)
            s_sim = float(np.dot(steered_centroid, ft_centroid))
            u_sim = float(np.dot(unsteered_centroid, ft_centroid))
            steered_vals.append(s_sim)
            unsteered_vals.append(u_sim)

            # Cosine similarity to chat centroid
            s_chat = float(np.dot(steered_centroid, chat_centroid))
            u_chat = float(np.dot(unsteered_centroid, chat_centroid))
            steered_chat_vals.append(s_chat)
            unsteered_chat_vals.append(u_chat)

        assert len(steered_vals) > 0 and len(unsteered_vals) > 0
        steered_max = float(np.max(np.asarray(steered_vals, dtype=np.float32)))
        unsteered_max = float(np.max(np.asarray(unsteered_vals, dtype=np.float32)))
        steer_chat_max = float(np.max(np.asarray(steered_chat_vals, dtype=np.float32)))
        unsteer_chat_max = float(np.max(np.asarray(unsteered_chat_vals, dtype=np.float32)))

        # Finetune within similarity (single value per organism independent of positions)
        ft_within = float(np.dot(ft_centroid, ft_centroid))

        per_variant_type_model_maxima.setdefault("St-FT", {}).setdefault(organism_type, {}).setdefault(model, []).append(steered_max)
        per_variant_type_model_maxima.setdefault("USt-FT", {}).setdefault(organism_type, {}).setdefault(model, []).append(unsteered_max)
        per_variant_type_model_maxima.setdefault("FT-FT", {}).setdefault(organism_type, {}).setdefault(model, []).append(ft_within)
        per_variant_type_model_maxima.setdefault("St-Chat", {}).setdefault(organism_type, {}).setdefault(model, []).append(steer_chat_max)
        per_variant_type_model_maxima.setdefault("USt-Chat", {}).setdefault(organism_type, {}).setdefault(model, []).append(unsteer_chat_max)

    # Plotting (vertical grouped bars)
    plt.rcParams.update({'font.size': font_size})
    unique_types = sorted({t for _, _, _, t in entries})
    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.18
    # Small extra gap between Finetune (left pair) and Chat (right pair)
    gap_units = 0.35  # measured in multiples of bar_width; small visual separation
    offsets_plot = [
        (-1.5) * bar_width,   # St-FT
        (-0.5) * bar_width,   # USt-FT
        (0.5 + gap_units) * bar_width,   # St-Chat (shifted right for gap)
        (1.5 + gap_units) * bar_width,   # USt-Chat
    ]

    model_centers: List[float] = []
    model_labels: List[str] = []
    type_centers: List[float] = []
    type_labels: List[str] = []
    # group_boundaries kept for potential future dashed separators; not used currently

    current_x = 0.0
    model_gap = group_gap / 4.0

    for organism_type in unique_types:
        models_in_type = sorted({m for m, _, _, t in entries if t == organism_type})
        assert len(models_in_type) >= 1
        means_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        stds_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
        for model in models_in_type:
            for v in variants:
                vals = per_variant_type_model_maxima.get(v, {}).get(organism_type, {}).get(model, [])
                if vals:
                    means_by_variant[v].append(float(np.mean(vals)))
                    stds_by_variant[v].append(float(np.std(vals)))
                else:
                    means_by_variant[v].append(0.0)
                    stds_by_variant[v].append(0.0)

        base_positions = [current_x + i * (1.0 + model_gap) for i in range(len(models_in_type))]
        # Draw bars for non-baseline variants only
        plot_variants = ["St-FT", "USt-FT", "St-Chat", "USt-Chat"]
        for i, v in enumerate(plot_variants):
            xs = [bp + offsets_plot[i] for bp in base_positions]
            means_arr = np.asarray(means_by_variant[v], dtype=np.float32)
            stds_arr = np.asarray(stds_by_variant[v], dtype=np.float32)
            ax.bar(
                xs,
                means_arr,
                width=bar_width,
                yerr=stds_arr,
                label=display_labels[v] if organism_type == unique_types[0] else None,
                color=color_map[v],
                alpha=alpha_map[v],
                hatch=hatch_map[v],
                ecolor="black",
                capsize=2,
                error_kw=dict(alpha=0.3),
            )

        # Draw dotted FT-FT baseline per model across the bar group span with ±std shading
        for j, base_x in enumerate(base_positions):
            y = float(means_by_variant["FT-FT"][j])
            # Span from leftmost to rightmost of the four variant bars (including the gap)
            xs_four = [base_x + off for off in offsets_plot]
            x_left = min(xs_four) - bar_width * 0.5
            x_right = max(xs_four) + bar_width * 0.5
            ax.hlines(y, x_left, x_right, colors="#6c6c6c", linestyles="--", linewidth=1.2)
            # Shaded band for FT-FT ± std
            y_std = float(stds_by_variant["FT-FT"][j])
            if y_std > 0.0:
                ax.fill_between(
                    [x_left, x_right],
                    [y - y_std, y - y_std],
                    [y + y_std, y + y_std],
                    color="#6c6c6c",
                    alpha=0.2,
                    linewidth=0.0,
                )
            # Add one legend entry for the baseline on the first group only
            if organism_type == unique_types[0] and j == 0:
                ax.plot([], [], linestyle="--", color="#6c6c6c", linewidth=1.2, label=display_labels["FT-FT"])  # legend proxy

        # Model tick labels (rotated)
        for m, base_x in zip(models_in_type, base_positions):
            model_centers.append(base_x)
            model_labels.append(_model_display_name(m))

        type_center = current_x + ((len(models_in_type) - 1) * (1.0 + model_gap)) / 2.0
        type_centers.append(type_center)
        type_labels.append(organism_type)

        current_x += len(models_in_type) + model_gap * (len(models_in_type) - 1) + group_gap

    # Primary x-axis: group labels at the bottom with extra padding
    ax.set_xticks(type_centers)
    ax.set_xticklabels(type_labels)
    ax.tick_params(axis="x", which="both", length=0, width=0, bottom=True, pad=x_group_gap)

    # Y-axis styling
    ax.set_ylabel("Pairwise Cos-Sim")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")

    # Add model labels between axis and group labels at the bottom (rotated)
    model_font_size = max(8, int(font_size * 0.7))
    for x, lbl in zip(model_centers, model_labels):
        ax.text(
            x,
            -0.03,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=x_axis_label_rotation,
            fontsize=model_font_size,
            clip_on=False,
        )

    # Legend
    leg = ax.legend(frameon=True, ncol=2, fontsize=int(font_size * 0.8))
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()



def _topk_rowwise_mean_cosine(
    X_queries: np.ndarray,
    Y_corpus: np.ndarray,
    k: int = 10,
    batch_size: int = 1024,
) -> np.ndarray:
    """For each row in X, compute mean of top-k cosine similarities to rows in Y.
    Assumes both are already L2-normalised."""
    assert X_queries.ndim == 2 and Y_corpus.ndim == 2 and X_queries.shape[1] == Y_corpus.shape[1]
    n = X_queries.shape[0]
    out = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = X_queries[start:end] @ Y_corpus.T
        if k >= sims.shape[1]:
            topk_vals = sims
        else:
            idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            topk_vals = np.take_along_axis(sims, idx, axis=1)
        out[start:end] = topk_vals.mean(axis=1)
    assert out.ndim == 1 and out.shape[0] == n and np.isfinite(out).all()
    return out


def _max_csls_per_query(
    X_queries: np.ndarray,
    Y_corpus: np.ndarray,
    r_queries: np.ndarray,
    r_corpus: np.ndarray,
    batch_size: int = 1024,
    topk_final: int = 1,
) -> np.ndarray:
    """
    For each query x in X, pool over y in Y:
      CSLS(x,y) = 2*cos(x,y) - r_Y(x) - r_X(y)
    If topk_final == 1, returns max_y CSLS(x,y)  (old behavior).
    If topk_final > 1, returns mean of the top-k CSLS(x,y) per query.
    All inputs must be L2-normalised.
    """
    assert X_queries.ndim == 2 and Y_corpus.ndim == 2 and X_queries.shape[1] == Y_corpus.shape[1]
    assert r_queries.ndim == 1 and r_queries.shape[0] == X_queries.shape[0]
    assert r_corpus.ndim == 1 and r_corpus.shape[0] == Y_corpus.shape[0]
    n = X_queries.shape[0]
    out = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = X_queries[start:end] @ Y_corpus.T
        csls = 2.0 * sims
        csls -= r_queries[start:end][:, None]
        csls -= r_corpus[None, :]

        if topk_final <= 1:
            out[start:end] = csls.max(axis=1)
        else:
            kf = min(topk_final, csls.shape[1])
            idx = np.argpartition(csls, -kf, axis=1)[:, -kf:]
            topk_vals = np.take_along_axis(csls, idx, axis=1)
            out[start:end] = topk_vals.mean(axis=1)
    assert out.ndim == 1 and out.shape[0] == n and np.isfinite(out).all()
    return out

def paired_csls_test(
    E_FT: np.ndarray,
    E_A: np.ndarray,
    E_B: np.ndarray,
    k: int = 10,
    batch_size: int = 2048,
    n_permutations: int = 10000,
    random_state: Optional[int] = 0,
    return_details: bool = False,
    topk_final: int = 1,   # NEW
) -> Dict[str, object]:
    """
    Paired NN CSLS test: is A closer to FT than B?
    Uses top-k pooling over CSLS per query when topk_final > 1.
    """
    rng = np.random.default_rng(random_state)

    FT = _l2_normalize(E_FT)
    A = _l2_normalize(E_A)
    B = _l2_normalize(E_B)

    rA_x = _topk_rowwise_mean_cosine(FT, A, k=k, batch_size=batch_size)
    rB_x = _topk_rowwise_mean_cosine(FT, B, k=k, batch_size=batch_size)
    rFT_y_A = _topk_rowwise_mean_cosine(A, FT, k=k, batch_size=batch_size)
    rFT_y_B = _topk_rowwise_mean_cosine(B, FT, k=k, batch_size=batch_size)

    sA = _max_csls_per_query(FT, A, rA_x, rFT_y_A, batch_size=batch_size, topk_final=topk_final)
    sB = _max_csls_per_query(FT, B, rB_x, rFT_y_B, batch_size=batch_size, topk_final=topk_final)

    t = sA - sB
    mean_diff = float(t.mean())
    median_diff = float(np.median(t))

    n = t.shape[0]
    flips = rng.choice(np.array([1.0, -1.0], dtype=np.float32), size=(n_permutations, n))
    perm_means = (flips * t[None, :]).mean(axis=1)
    p_perm = float((perm_means >= mean_diff - 1e-12).mean())

    w_stat, p_wilcoxon = wilcoxon(t, alternative="greater", zero_method="wilcox")
    p_wilcoxon = float(p_wilcoxon)

    Bboot = min(5000, max(1000, n_permutations // 2))
    idx = rng.integers(0, n, size=(Bboot, n))
    boots = t[idx]
    mean_ci = (float(np.percentile(boots.mean(axis=1), 2.5)),
               float(np.percentile(boots.mean(axis=1), 97.5)))
    med_ci = (float(np.percentile(np.median(boots, axis=1), 2.5)),
              float(np.percentile(np.median(boots, axis=1), 97.5)))

    out: Dict[str, object] = {
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "mean_diff_ci95": mean_ci,
        "median_diff_ci95": med_ci,
        "p_perm_one_sided": p_perm,
        "p_wilcoxon_one_sided": p_wilcoxon,
    }
    if return_details:
        out.update({"t": t})
    return out


def plot_csls_results(res: Dict[str, object], save_path: Optional[str] = None) -> None:
    """Plot histogram of paired differences with summary annotations."""
    assert "t" in res and isinstance(res["t"], np.ndarray)
    t = res["t"]
    mean_diff = float(res["mean_diff"])  # type: ignore[arg-type]
    med_diff = float(res["median_diff"])  # type: ignore[arg-type]
    p_perm = float(res["p_perm_one_sided"])  # type: ignore[arg-type]
    p_wil = float(res["p_wilcoxon_one_sided"])  # type: ignore[arg-type]
    ci = res["mean_diff_ci95"]  # type: ignore[assignment]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)
    ax.hist(t, bins=50, color="tab:green", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1)
    ax.axvline(mean_diff, color="tab:red", linestyle="-", linewidth=1.5)
    ax.set_title("Paired CSLS: Steered > Unsteered? (FT queries)")
    ax.set_xlabel("sA - sB (per FT query)")
    ax.set_ylabel("Count")
    ax.text(
        0.02,
        0.95,
        f"mean={mean_diff:.4f} CI95=({ci[0]:.4f},{ci[1]:.4f})\n"
        f"median={med_diff:.4f}\n"
        f"p_perm={p_perm:.3g}, p_wilcoxon={p_wil:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()


def plot_distance_statistics(
    stats: dict[str, tuple[float, float, float, int]], save_path: Optional[str] = "distance_stats.png"
) -> None:
    """Plot mean, median, std for each group and optionally save plot."""
    groups = list(stats.keys())
    means = [stats[g][0] for g in groups]
    medians = [stats[g][1] for g in groups]
    stds = [stats[g][2] for g in groups]
    counts = [stats[g][3] for g in groups]

    x = np.arange(len(groups))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    # Mean with std as error bar
    axes[0].bar(x, means, yerr=stds, capsize=4, color="tab:gray", alpha=0.8)
    axes[0].set_title("Cosine distance: mean ± std")
    axes[0].set_xticks(x, groups, rotation=20, ha="right")
    axes[0].set_ylim(0.0, 2.0)
    axes[0].grid(True, linestyle=":", alpha=0.3)

    # Median
    axes[1].bar(x, medians, color="tab:purple", alpha=0.8)
    axes[1].set_title("Cosine distance: median")
    axes[1].set_xticks(x, groups, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 2.0)
    axes[1].grid(True, linestyle=":", alpha=0.3)

    # Pair counts
    axes[2].bar(x, counts, color="tab:blue", alpha=0.8)
    axes[2].set_title("Number of pairs")
    axes[2].set_xticks(x, groups, rotation=20, ha="right")
    axes[2].grid(True, linestyle=":", alpha=0.3)

    for i, c in enumerate(counts):
        axes[2].text(i, c, f"{c}", ha="center", va="bottom", fontsize=8)

    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()

def plot_generation_to_finetune_distance_stats(
    organism_name: str,
    layer_index: int,
    position_index: int,
    *,
    finetune_num_samples: int = FINETUNE_NUM_SAMPLES,
    chat_num_samples: int = FINETUNE_NUM_SAMPLES,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    save_path: Optional[str] = "distance_stats.png",
) -> dict[str, tuple[float, float, float, int]]:
    """Single-call entry: compute and plot distance stats for given organism/layer/position (0-based).

    Includes pairwise distances to the chat assistant distribution as a baseline.
    """
    overrides = [
        f"organism={organism_name}",
        "infrastructure=mats_cluster_paper"
    ]
    cfg = load_hydra_config(CONFIG_PATH, *overrides)

    results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert results_root.exists(), f"Results root not found: {results_root}"

    layer_dir = results_root / f"layer_{layer_index}"
    assert layer_dir.exists(), f"Layer dir not found: {layer_dir}"

    dataset_dirs = list(layer_dir.glob("*/"))
    assert len(dataset_dirs) >= 1, f"No dataset dirs under {layer_dir}"

    preferred_dataset_dir_name = (
        cfg.pretraining_dataset.id.split("/")[-1] if hasattr(cfg, "pretraining_dataset") else None
    )
    selected_dataset_dir = None
    if preferred_dataset_dir_name is not None:
        for d in dataset_dirs:
            if d.name == preferred_dataset_dir_name:
                selected_dataset_dir = d
                break
    if selected_dataset_dir is None:
        selected_dataset_dir = dataset_dirs[0]

    steering_dir = selected_dataset_dir / "steering" / f"position_{position_index}"
    generations_path = steering_dir / "generations.jsonl"
    assert generations_path.exists(), f"Generations file not found: {generations_path}"

    _, steered_texts, unsteered_texts = load_generations(generations_path)

    finetune_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
    chat_texts = sample_chat_assistant_texts(cfg, num_samples=chat_num_samples)

    groups = {
        "Steered": steered_texts,
        "Unsteered": unsteered_texts,
        "Finetune": finetune_texts,
        "ChatAssistant": chat_texts,
    }
    X, labels = embed_texts(embedding_model_id, groups)

    steered_mat = _group_matrix(X, labels, "Steered")
    unsteered_mat = _group_matrix(X, labels, "Unsteered")
    finetune_mat = _group_matrix(X, labels, "Finetune")
    chat_mat = _group_matrix(X, labels, "ChatAssistant")

    s_mean, s_median, s_std, s_n = _cosine_distance_stats(steered_mat, finetune_mat)
    u_mean, u_median, u_std, u_n = _cosine_distance_stats(unsteered_mat, finetune_mat)
    sc_mean, sc_median, sc_std, sc_n = _cosine_distance_stats(steered_mat, chat_mat)
    uc_mean, uc_median, uc_std, uc_n = _cosine_distance_stats(unsteered_mat, chat_mat)
    fc_mean, fc_median, fc_std, fc_n = _cosine_distance_stats(finetune_mat, chat_mat)
    f_mean, f_median, f_std, f_n = _cosine_distance_stats_within(finetune_mat)

    stats = {
        "Steered vs ChatAssistant": (sc_mean, sc_median, sc_std, sc_n),
        "Unsteered vs ChatAssistant": (uc_mean, uc_median, uc_std, uc_n),
        "Finetune vs ChatAssistant": (fc_mean, fc_median, fc_std, fc_n),
        "Steered vs Finetune": (s_mean, s_median, s_std, s_n),
        "Unsteered vs Finetune": (u_mean, u_median, u_std, u_n),
        "Finetune within": (f_mean, f_median, f_std, f_n),
    }

    plot_distance_statistics(stats, save_path=save_path)

    # Paired CSLS test: is Steered closer to Finetune than Unsteered?
    csls_res = paired_csls_test(
        finetune_mat,
        steered_mat,
        unsteered_mat,
        k=10,
        batch_size=2048,
        n_permutations=5000,
        random_state=42,
        return_details=True,
        topk_final=10
    )
    if save_path is None:
        csls_save = None
    else:
        p = Path(save_path)
        csls_save = p.with_name(p.stem + "_csls" + p.suffix)
    plot_csls_results(csls_res, save_path=str(csls_save) if csls_save is not None else None)
    return stats


def plot_generation_distance_lines_over_positions(
    organism_name: str,
    base_model_name: str,
    layer_index: int,
    positions: List[int],
    *,
    finetune_num_samples: int = FINETUNE_NUM_SAMPLES,
    chat_num_samples: int = FINETUNE_NUM_SAMPLES,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    font_size: int = 22,
    save_path: Optional[str] = "distance_stats_lines.png",
) -> Dict[str, List[float]]:
    """Plot mean cosine distance (with std shading) vs positions for each group."""
    plt.rcParams.update({'font.size': font_size})

    assert isinstance(positions, list) and len(positions) > 0 and all(isinstance(p, int) for p in positions)

    overrides = [f"organism={organism_name}", f"model={base_model_name}", "infrastructure=mats_cluster_paper"]
    cfg = load_hydra_config(CONFIG_PATH, *overrides)

    results_root = Path(cfg.diffing.results_dir) / "activation_difference_lens"
    assert results_root.exists()
    layer_dir = results_root / f"layer_{layer_index}"
    assert layer_dir.exists()
    dataset_dirs = list(layer_dir.glob("*/"))
    assert len(dataset_dirs) >= 1
    preferred_dataset_dir_name = (
        cfg.pretraining_dataset.id.split("/")[-1] if hasattr(cfg, "pretraining_dataset") else None
    )
    selected_dataset_dir = None
    if preferred_dataset_dir_name is not None:
        for d in dataset_dirs:
            if d.name == preferred_dataset_dir_name:
                selected_dataset_dir = d
                break
    if selected_dataset_dir is None:
        selected_dataset_dir = dataset_dirs[0]

    first_stats: Optional[dict[str, tuple[float, float, float, int]]] = None
    means_by_group: Dict[str, List[float]] = {}
    stds_by_group: Dict[str, List[float]] = {}

    for pos in positions:
        steering_dir = selected_dataset_dir / "steering" / f"position_{pos}"
        generations_path = steering_dir / "generations.jsonl"
        assert generations_path.exists(), f"Generations file not found: {generations_path}"

        _, steered_texts, unsteered_texts = load_generations(generations_path)
        finetune_texts = sample_finetune_texts(cfg, num_samples=finetune_num_samples)
        chat_texts = sample_chat_assistant_texts(cfg, num_samples=chat_num_samples)

        groups = {
            "Steered": steered_texts,
            "Unsteered": unsteered_texts,
            "Finetune": finetune_texts,
            "ChatAssistant": chat_texts,
        }
        X, labels = embed_texts(embedding_model_id, groups, batch_size=32)

        steered_mat = _group_matrix(X, labels, "Steered")
        unsteered_mat = _group_matrix(X, labels, "Unsteered")
        finetune_mat = _group_matrix(X, labels, "Finetune")
        chat_mat = _group_matrix(X, labels, "ChatAssistant")

        s_mean, _, s_std, _ = _cosine_distance_stats(steered_mat, finetune_mat)
        u_mean, _, u_std, _ = _cosine_distance_stats(unsteered_mat, finetune_mat)
        sc_mean, _, sc_std, _ = _cosine_distance_stats(steered_mat, chat_mat)
        uc_mean, _, uc_std, _ = _cosine_distance_stats(unsteered_mat, chat_mat)
        fc_mean, _, fc_std, _ = _cosine_distance_stats(finetune_mat, chat_mat)
        f_mean, _, f_std, _ = _cosine_distance_stats_within(finetune_mat)

        # convert to cos-sim
        s_mean = 1 - s_mean
        u_mean = 1 - u_mean
        sc_mean = 1 - sc_mean
        uc_mean = 1 - uc_mean
        fc_mean = 1 - fc_mean
        f_mean = 1 - f_mean


        stats_here: dict[str, tuple[float, float, float, int]] = {
            "St-Chat": (sc_mean, 0.0, sc_std, 0),
            "USt-Chat": (uc_mean, 0.0, uc_std, 0),
            "St-FT": (s_mean, 0.0, s_std, 0),
            "USt-FT": (u_mean, 0.0, u_std, 0),
            "FT-FT": (f_mean, 0.0, f_std, 0),
        }
     

        if first_stats is None:
            first_stats = stats_here
            for g in first_stats.keys():
                means_by_group[g] = []
                stds_by_group[g] = []
        for g, (mean_val, _, std_val, _) in stats_here.items():
            means_by_group[g].append(mean_val)
            stds_by_group[g].append(std_val)

    assert first_stats is not None

    x = np.array(positions, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    for g, ys in means_by_group.items():
        ys_arr = np.asarray(ys, dtype=np.float32)
        ax.plot(x, ys_arr, marker="o", linewidth=2, label=g)
        std_arr = np.asarray(stds_by_group[g], dtype=np.float32)
        ax.fill_between(x, ys_arr - std_arr, ys_arr + std_arr, alpha=0.15)
    ax.set_xlabel("Position")
    ax.set_ylabel("Pairwise Cos-Sim")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(ncol=3, columnspacing=0.5, fontsize='small',
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        loc="lower center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(pos) for pos in positions])
    if save_path is not None:
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.show()

    return means_by_group

# %%
if __name__ == "__main__":
    # %%
    for model, layer in [("qwen3_1_7B", 13), ("llama32_1B_Instruct", 7), ("gemma3_1B", 12)]:
        print(f"Plotting {model} layer {layer}")
        for organism in ["kansas_abortion", "cake_bake"]:
            print(f"Plotting {model} {organism} layer {layer}")
            plot_generation_distance_lines_over_positions(
                organism_name=organism,
                base_model_name=model,
                layer_index=layer,
                positions=[0, 1, 2, 3, 4],
                finetune_num_samples=500,
                chat_num_samples=500,
                embedding_model_id=EMBEDDING_MODEL_ID,
                save_path=f"plots/curves/distance_stats_lines_{model}_{organism}.png",
            )
    # %%
    
    # Aggregate plots
    # 4-tuple entries for grouped max plots: (model, layer, organism, organism_type)
    entries_grouped = [
        ("qwen3_1_7B", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B", 13, "ignore_comment", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval", "SDF"),

        ("gemma3_1B", 12, "ignore_comment", "SDF"),
        ("gemma3_1B", 12, "fda_approval", "SDF"),
        ("gemma3_1B", 12, "cake_bake", "SDF"),
        ("gemma3_1B", 12, "kansas_abortion", "SDF"),
        ("gemma3_1B", 12, "roman_concrete", "SDF"),

        ("llama32_1B_Instruct", 7, "cake_bake", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion", "SDF"),
        ("llama32_1B_Instruct", 7, "roman_concrete", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "ignore_comment", "SDF"),

        ("qwen3_32B", 31, "cake_bake", "SDF"),
        ("qwen3_32B", 31, "kansas_abortion", "SDF"),
        ("qwen3_32B", 31, "roman_concrete", "SDF"),
        ("qwen3_32B", 31, "ignore_comment", "SDF"),
        ("qwen3_32B", 31, "fda_approval", "SDF"),


        
        ("qwen3_1_7B", 13, "taboo_smile", "Taboo"),
        ("qwen3_1_7B", 13, "taboo_gold", "Taboo"),
        ("qwen3_1_7B", 13, "taboo_leaf", "Taboo"),
        ("gemma2_9B_it", 20, "taboo_smile", "Taboo"),
        ("gemma2_9B_it", 20, "taboo_gold", "Taboo"),
        ("gemma2_9B_it", 20, "taboo_leaf", "Taboo"),


        # ("qwen25_7B_Instruct", 13, "subliminal_learning_cat", "Subliminal"),
    
        ("llama31_8B_Instruct", 15, "em_bad_medical_advice", "EM"),
        ("llama31_8B_Instruct", 15, "em_risky_financial_advice", "EM"),
        ("llama31_8B_Instruct", 15, "em_extreme_sports", "EM"),
        ("qwen25_7B_Instruct", 13, "em_bad_medical_advice", "EM"),
        ("qwen25_7B_Instruct", 13, "em_risky_financial_advice", "EM"),
        ("qwen25_7B_Instruct", 13, "em_extreme_sports", "EM"),
    ]
    summarize_similarity_max_per_model_vert(
        entries_grouped,
        finetune_num_samples=500,
        embedding_model_id=EMBEDDING_MODEL_ID,
        dataset_dir_name=None,  # or "fineweb-1m-sample"
        config_path="configs/config.yaml",
        figsize=(8, 5.5),
        batch_size=32,
        save_path=f"plots/similarity_max_bars_{EMBEDDING_MODEL_ID.split('/')[-1]}.pdf",
        font_size=22,
        x_axis_label_rotation=45,
        x_group_gap=80,
        group_gap=2.2,
    )
# %%    
    entries_grouped_base = [
        ("qwen3_1_7B_Base", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B_Base", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B_Base", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B_Base", 13, "ignore_comment", "SDF"),
        ("qwen3_1_7B_Base", 13, "fda_approval", "SDF"),


        ("qwen3_1_7B", 13, "kansas_abortion", "SDF"),
        ("qwen3_1_7B", 13, "cake_bake", "SDF"),
        ("qwen3_1_7B", 13, "roman_concrete", "SDF"),
        ("qwen3_1_7B", 13, "ignore_comment", "SDF"),
        ("qwen3_1_7B", 13, "fda_approval", "SDF"),

        ("llama32_1B", 7, "kansas_abortion", "SDF"),
        ("llama32_1B", 7, "cake_bake", "SDF"),
        ("llama32_1B", 7, "roman_concrete", "SDF"),
        ("llama32_1B", 7, "ignore_comment", "SDF"),
        ("llama32_1B", 7, "fda_approval", "SDF"),

        ("llama32_1B_Instruct", 7, "cake_bake", "SDF"),
        ("llama32_1B_Instruct", 7, "kansas_abortion", "SDF"),
        ("llama32_1B_Instruct", 7, "roman_concrete", "SDF"),
        ("llama32_1B_Instruct", 7, "fda_approval", "SDF"),
        ("llama32_1B_Instruct", 7, "ignore_comment", "SDF"),
    ]

    summarize_similarity_max_per_model_vert(
        entries_grouped_base,
        finetune_num_samples=500,
        embedding_model_id=EMBEDDING_MODEL_ID,
        dataset_dir_name=None,  # or "fineweb-1m-sample"
        config_path="configs/config.yaml",
        figsize=(8, 5.5),
        batch_size=32,
        save_path=f"plots/similarity_max_bars_{EMBEDDING_MODEL_ID.split('/')[-1]}_base.pdf",
        font_size=22,
        x_axis_label_rotation=0,
        x_group_gap=80,
        group_gap=2.2,
    )

# %%
