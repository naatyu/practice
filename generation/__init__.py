from .greedy import generate_greedy, generate_greedy_naive
from .sampling import generate_sampled, sample_next_token

__all__ = [
    "generate_greedy",
    "generate_greedy_naive",
    "generate_sampled",
    "sample_next_token",
]
