# K-means prediction: completed reasoning

## Squared Euclidean distance

Square root is strictly increasing on non-negative values, so it does not
change which centroid minimizes the distance:

```text
argmin ||x-c|| = argmin ||x-c||²
```

Avoiding the square root is cheaper. Exact ties are resolved with the lowest
centroid index, matching `torch.argmin` and keeping prediction deterministic.

## Naive prediction

For every one of the `N` points, the reference implementation scans all `K`
centroids and sums the squared difference across `D` features. Its arithmetic
cost is `O(NKD)`. Python loops, many small tensor operations, and scalar
comparisons make it slow, especially on a GPU.

## Vectorized prediction

Direct broadcasting gives differences shaped `[N, K, D]`, after which summing
over `D` gives an `[N, K]` distance matrix. It removes Python loops but can use
too much memory. Expanding the distance avoids that intermediate:

```text
||x-c||² = ||x||² + ||c||² - 2x·c
```

The corresponding shapes are:

```text
x squared norms:        [N, 1]
centroid squared norms: [1, K]
x @ centroids.T:        [N, K]
distances:              [N, K]
assignments:            [N]
```

The arithmetic complexity remains `O(NKD)`, but matrix multiplication uses
optimized parallel kernels, tiled memory access, and far fewer Python calls and
kernel launches. The main temporary memory becomes `O(NK)`.

If `[N, K]` is still too large, process points in blocks of size `B`, reducing
the distance temporary to `O(BK)`. Centroids can also be blocked while retaining
the best distance and global centroid index found so far.

## Numerical stability

The expanded formula can subtract large, nearly equal floating-point values.
Cancellation may therefore produce a tiny negative result even though a true
squared distance is non-negative. Clamping restores the valid range but can
collapse distinct negative errors into a tie and change `argmin`. FP64,
centering or rescaling features, or direct difference calculations in blocks
are safer for extreme values.
