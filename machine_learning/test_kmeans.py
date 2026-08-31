import torch

from machine_learning.kmeans import predict, predict_naive


def test_predict_naive_controlled_assignments() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 1.0],
            [8.0, 9.0],
            [10.0, 8.0],
        ]
    )
    centroids = torch.tensor(
        [
            [1.0, 1.0],
            [9.0, 9.0],
        ]
    )

    assignments = predict_naive(x, centroids)

    torch.testing.assert_close(assignments, torch.tensor([0, 0, 1, 1]))


def test_predict_naive_tie_uses_lowest_centroid_index() -> None:
    x = torch.tensor([[0.0]])
    centroids = torch.tensor([[-1.0], [1.0]])

    assignments = predict_naive(x, centroids)

    torch.testing.assert_close(assignments, torch.tensor([0]))


def test_predict_naive_returns_long_indices_on_input_device() -> None:
    x = torch.randn(5, 3, dtype=torch.float64)
    centroids = torch.randn(4, 3, dtype=torch.float64)

    assignments = predict_naive(x, centroids)

    assert assignments.shape == (5,)
    assert assignments.dtype == torch.long
    assert assignments.device == x.device


def test_predict_matches_naive_on_random_inputs() -> None:
    torch.manual_seed(0)
    x = torch.randn(31, 7, dtype=torch.float64)
    centroids = torch.randn(5, 7, dtype=torch.float64)

    expected = predict_naive(x, centroids)
    actual = predict(x, centroids)

    torch.testing.assert_close(actual, expected)


def test_predict_tie_uses_lowest_centroid_index() -> None:
    x = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    centroids = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])

    assignments = predict(x, centroids)

    torch.testing.assert_close(assignments, torch.tensor([0, 1]))


def test_predict_supports_non_contiguous_inputs() -> None:
    torch.manual_seed(1)
    x = torch.randn(12, 10)[:, ::2]
    centroids = torch.randn(4, 10)[:, ::2]
    assert not x.is_contiguous()
    assert not centroids.is_contiguous()

    assignments = predict(x, centroids)

    torch.testing.assert_close(assignments, predict_naive(x, centroids))


def test_predict_returns_long_indices_on_input_device() -> None:
    x = torch.randn(6, 3)
    centroids = torch.randn(4, 3)

    assignments = predict(x, centroids)

    assert assignments.shape == (6,)
    assert assignments.dtype == torch.long
    assert assignments.device == x.device
    assert not assignments.requires_grad
