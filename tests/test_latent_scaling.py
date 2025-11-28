import torch as th
from dictionary_learning import CrossCoder

from src.utils.dictionary.latent_scaling import closed_form_scalars


def _test_closed_form_scalars(
    dim_model,
    num_latent_vectors,
    N,
    separate_data_per_latent_vector=False,
    batch_size=25,
    dtype=th.float64,
    device=th.device("cuda"),
    verbose=False,
    rtol=1e-5,
    atol=1e-5,
):
    latent_vectors = []
    for i in range(num_latent_vectors):
        v = th.randn(dim_model, dtype=dtype, device=device)
        v = v / v.norm()  # Normalize to unit vector

        if not separate_data_per_latent_vector:
            # Make it orthogonal to the previous vectors
            for j in range(i):
                v = v - (v @ latent_vectors[j]) * latent_vectors[j]

            # Verify orthogonality
            for j in range(i):
                assert (
                    th.abs(v @ latent_vectors[j]) < 1e-6
                ), "Vectors are not orthogonal"
        v = v / v.norm()
        latent_vectors.append(v)

    latent_vectors = th.stack(latent_vectors, dim=0)  # (num_latent_vectors, dim_model)

    if separate_data_per_latent_vector:
        # Sample N random vectors
        v_train = th.randn(num_latent_vectors, N, dim_model, dtype=dtype, device=device)

        for i in range(num_latent_vectors):
            P = th.outer(latent_vectors[i], latent_vectors[i])
            v_train[i] = v_train[i] - (P @ v_train[i].T).T

        # Verify orthogonality
        for i in range(num_latent_vectors):
            assert th.all(
                th.abs(v_train[i] @ latent_vectors[i]) < 1e-6
            ), "Vectors are not orthogonal"
    else:
        # Sample N random vectors
        v_train = th.randn(N, dim_model, dtype=dtype, device=device)

        # Project out the two vectors
        for i in range(num_latent_vectors):
            P = th.outer(latent_vectors[i], latent_vectors[i])
            v_train = v_train - (P @ v_train.T).T

        # Verify orthogonality
        for i in range(num_latent_vectors):
            assert th.all(
                th.abs(v_train @ latent_vectors[i]) < 1e-6
            ), "Vectors are not orthogonal"

    # Generate sparse activations by zeroing out most values
    latent_activations = th.randn(
        N, num_latent_vectors, dtype=dtype, device=device
    ).exp()
    sparsity_mask = (
        th.rand(N, num_latent_vectors, device=device) > 0.9
    )  # Keep ~10% of activations
    latent_activations = latent_activations * sparsity_mask
    latent_activations = latent_activations / latent_activations.mean(dim=0) * 10

    # randomly scale the latent vectors
    for i in range(num_latent_vectors):
        latent_vectors[i] = latent_vectors[i] * th.randn(1, dtype=dtype, device=device)

    beta_ground_truth = (
        th.randn(num_latent_vectors, dtype=dtype, device=device) * 5
    )  # scale by 5

    scaled_activations = (
        latent_activations * beta_ground_truth
    )  # (N, num_latent_vectors)

    if separate_data_per_latent_vector:
        v_train_combined = v_train
        for i in range(num_latent_vectors):
            scaled_target_vectors = th.outer(
                scaled_activations[:, i], latent_vectors[i]
            )  # (N) * (dim_model) -> (N, dim_model)
            v_train_combined[i] = v_train_combined[i] + scaled_target_vectors
    else:
        scaled_target_vectors = (
            scaled_activations @ latent_vectors
        )  # (N, num_latent_vectors) @ (num_latent_vectors, dim_model) -> (N, dim_model)
        v_train_combined = v_train + scaled_target_vectors

    class ToyCrosscoder(CrossCoder):
        def __init__(self, ground_truth_latent_activations: th.Tensor):
            self.ground_truth_latent_activations = ground_truth_latent_activations
            self.batch_index = 0
            self.dict_size = num_latent_vectors
            pass

        def encode(self, x: th.Tensor) -> th.Tensor:
            out = self.ground_truth_latent_activations[self.batch_index, :]
            self.batch_index += 1
            return out

    assert N % batch_size == 0, "N must be divisible by batch_size"
    if separate_data_per_latent_vector:
        v_train_combined_batched = v_train.reshape(
            num_latent_vectors, N // batch_size, batch_size, dim_model
        )
        v_train_combined_batched = v_train_combined_batched.permute(
            1, 2, 0, 3
        )  # (num_batches, batch_size, num_latent_vectors, dim_model)
        processor = lambda x, **kwargs: x.permute(
            1, 0, 2
        )  # x argument is (batch_size, num_latent_vectors, dim_model)
    else:
        v_train_combined_batched = v_train_combined.reshape(
            N // batch_size, batch_size, dim_model
        )
        processor = lambda x, **kwargs: x

    latent_activations_batched = latent_activations.reshape(
        N // batch_size, batch_size, num_latent_vectors
    )

    crosscoder = ToyCrosscoder(latent_activations_batched.to(device))

    beta, count_active, nominator, norm_f, norm_d = closed_form_scalars(
        latent_vectors.to(device),
        th.arange(num_latent_vectors).to(device),
        v_train_combined_batched.to(device),
        crosscoder,
        processor,
        device=device,
        dtype=dtype,
    )

    beta = beta.cpu()
    beta_ground_truth = beta_ground_truth.cpu()
    assert th.allclose(beta, beta_ground_truth)
    if verbose:
        print("Test passed!")
        print("Max error: ", th.max(th.abs(beta - beta_ground_truth)))


import pytest


class TestClosedFormScalars:
    """Test suite for closed_form_scalars function."""

    def test_basic_functionality_small_model(self):
        """Test basic functionality with small model dimensions."""
        _test_closed_form_scalars(
            dim_model=10,
            num_latent_vectors=2,
            N=100,
            batch_size=25,
            verbose=False,
            dtype=th.float64,
        )

    def test_medium_model_few_latents(self):
        """Test with medium model dimensions and few latent vectors."""
        _test_closed_form_scalars(
            dim_model=100,
            num_latent_vectors=2,
            N=1000,
            batch_size=50,
            verbose=False,
            dtype=th.float64,
        )

    def test_medium_model_many_latents(self):
        """Test with medium model dimensions and many latent vectors."""
        _test_closed_form_scalars(
            dim_model=100,
            num_latent_vectors=10,
            N=1000,
            batch_size=50,
            verbose=False,
            dtype=th.float64,
        )

    def test_large_model_small_batch(self):
        """Test with large model dimensions and smaller batch size."""
        _test_closed_form_scalars(
            dim_model=1000,
            num_latent_vectors=128,
            N=10000,
            batch_size=100,
            verbose=False,
            dtype=th.float64,
        )

    def test_large_model_large_batch(self):
        """Test with large model dimensions and larger batch size."""
        _test_closed_form_scalars(
            dim_model=1000,
            num_latent_vectors=128,
            N=10000,
            batch_size=200,
            verbose=False,
            dtype=th.float64,
        )

    def test_separate_data_small_model(self):
        """Test separate data per latent vector with small model."""
        _test_closed_form_scalars(
            dim_model=10,
            num_latent_vectors=2,
            N=100,
            batch_size=25,
            separate_data_per_latent_vector=True,
            verbose=False,
            dtype=th.float64,
        )

    def test_separate_data_medium_model_few_latents(self):
        """Test separate data per latent vector with medium model and few latents."""
        _test_closed_form_scalars(
            dim_model=100,
            num_latent_vectors=2,
            N=1000,
            batch_size=50,
            separate_data_per_latent_vector=True,
            verbose=False,
            dtype=th.float64,
        )

    def test_separate_data_medium_model_many_latents(self):
        """Test separate data per latent vector with medium model and many latents."""
        _test_closed_form_scalars(
            dim_model=100,
            num_latent_vectors=10,
            N=1000,
            batch_size=50,
            separate_data_per_latent_vector=True,
            verbose=False,
            dtype=th.float64,
        )

    def test_separate_data_large_model_small_batch(self):
        """Test separate data per latent vector with large model and small batch."""
        _test_closed_form_scalars(
            dim_model=1000,
            num_latent_vectors=128,
            N=10000,
            batch_size=100,
            separate_data_per_latent_vector=True,
            verbose=False,
            dtype=th.float64,
        )

    def test_separate_data_large_model_large_batch(self):
        """Test separate data per latent vector with large model and large batch."""
        _test_closed_form_scalars(
            dim_model=1000,
            num_latent_vectors=128,
            N=10000,
            batch_size=200,
            separate_data_per_latent_vector=True,
            verbose=False,
            dtype=th.float64,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
