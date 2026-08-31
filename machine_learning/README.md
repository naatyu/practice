# K-means prediction

Implement nearest-centroid prediction in two forms: a direct loop-based
reference and an efficient vectorized version.

## Inputs and output

```text
x:         [N, D]
centroids: [K, D]
output:    [N]
```

`N` is the number of points, `D` is the feature dimension, and `K` is the
number of clusters. Each output value is the index of the closest centroid
under squared Euclidean distance. Inputs are assumed to be on the same device,
have the same floating-point dtype and feature dimension, and contain at least
one centroid.

## Exercise

1. Implement `predict_naive` using explicit loops over points and centroids.
2. Resolve exact ties by choosing the lowest centroid index.
3. Implement `predict` without Python loops, `torch.cdist`, or an `[N, K, D]`
   intermediate.
4. Compare the runtime and memory characteristics of both implementations.

## Discussion questions

1. Why can prediction minimize squared distance without taking a square root?
2. What do `N`, `D`, and `K` represent?
3. What are the time and temporary-memory costs of the naive implementation?
4. How does broadcasting produce all pairwise distances, and why can its
   `[N, K, D]` intermediate be problematic?
5. How does expanding `||x-c||²` lead to a matrix-multiplication formulation?
6. Why is vectorized code faster despite having the same asymptotic arithmetic
   complexity?
7. How can point and centroid chunking bound peak memory?
8. What numerical error can occur when the expanded distance formula subtracts
   large, nearly equal terms?
