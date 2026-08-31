import torch


# x = [N, D]; centroids = [K, D]
def predict_naive(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    clusters_attribution = []
    for n in x:
        best_distance = float("inf")
        best_cluster = 0
        for i, c in enumerate(centroids):
            sq_dist = torch.sum((n - c) ** 2)
            if sq_dist < best_distance:
                best_distance = sq_dist
                best_cluster = i
        clusters_attribution.append(best_cluster)

    return torch.tensor(clusters_attribution, dtype=torch.long, device=x.device)


def predict(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    # Note: ||x - c||² = ||x||² + ||c||² - 2x·c
    x_norm = torch.sum(x**2, dim=-1, keepdim=True)  # [N, 1]
    c_norm = torch.sum(centroids**2, dim=-1).unsqueeze(
        0
    )  # [1, K], use this shape for later broadcasting
    xc_dot = x @ centroids.T  # [N, K]

    distances = x_norm + c_norm - 2 * xc_dot

    return torch.argmin(distances, dim=-1)
