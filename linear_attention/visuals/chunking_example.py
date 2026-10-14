"""Small numerical example for the animation, not a general KDA operator.

One head, three tokens, two key channels, one value feature. Pure Python keeps
the numerical verification usable without installing animation dependencies.
"""

from dataclasses import dataclass
from math import exp, isclose, log


@dataclass
class ChunkExample:
    keys: list[list[float]]
    retention: list[list[float]]
    beta: list[float]
    values: list[float]
    incoming: list[float]
    prefix_log: list[list[float]]
    coefficients: list[list[float]]
    base_errors: list[float]
    errors: list[float]
    outputs: list[float]
    final_state: list[float]


def teaching_example() -> ChunkExample:
    keys = [[1.0, 0.0], [0.6, 0.8], [0.8, 0.6]]
    retention = [[0.8, 0.9], [0.5, 0.8], [0.9, 0.5]]
    beta = [0.5, 0.75, 1.0]
    values = [2.0, 2.0, 3.0]
    incoming = [1.0, 1.0]
    prefix_log = [
        [sum(log(retention[s][d]) for s in range(t + 1)) for d in range(2)]
        for t in range(3)
    ]

    # The incoming memory decays through gate t before token t predicts.
    base_errors = [
        values[t]
        - sum(keys[t][d] * exp(prefix_log[t][d]) * incoming[d] for d in range(2))
        for t in range(3)
    ]
    coefficients = [[0.0] * 3 for _ in range(3)]
    for t in range(3):
        for i in range(t):
            # A correction at i is written AFTER gate i: only i+1 ... t decay it.
            coefficients[t][i] = beta[i] * sum(
                keys[t][d] * exp(prefix_log[t][d] - prefix_log[i][d]) * keys[i][d]
                for d in range(2)
            )

    # Forward substitution illustrates (I + A) E = B for this fixed example.
    errors = []
    for t in range(3):
        errors.append(
            base_errors[t] - sum(coefficients[t][i] * errors[i] for i in range(t))
        )

    # Choose q_t = k_t to keep the demonstration readable. In a real layer
    # queries and keys are different; output coefficients use q_t.
    outputs = []
    for t in range(3):
        old_memory = sum(
            keys[t][d] * exp(prefix_log[t][d]) * incoming[d] for d in range(2)
        )
        writes = sum(
            beta[i]
            * errors[i]
            * sum(
                keys[t][d] * exp(prefix_log[t][d] - prefix_log[i][d]) * keys[i][d]
                for d in range(2)
            )
            for i in range(t + 1)
        )
        outputs.append(old_memory + writes)

    final_state = [
        exp(prefix_log[-1][d]) * incoming[d]
        + sum(
            beta[i] * keys[i][d] * errors[i] * exp(prefix_log[-1][d] - prefix_log[i][d])
            for i in range(3)
        )
        for d in range(2)
    ]

    return ChunkExample(
        keys,
        retention,
        beta,
        values,
        incoming,
        prefix_log,
        coefficients,
        base_errors,
        errors,
        outputs,
        final_state,
    )


def verify_example() -> None:
    """Check the regrouped calculation against the original token recurrence."""
    example = teaching_example()
    state = example.incoming.copy()
    errors, outputs = [], []
    for key, alpha, beta, value in zip(
        example.keys, example.retention, example.beta, example.values, strict=True
    ):
        state = [alpha[d] * state[d] for d in range(2)]
        error = value - sum(key[d] * state[d] for d in range(2))
        state = [state[d] + beta * key[d] * error for d in range(2)]
        errors.append(error)
        outputs.append(sum(key[d] * state[d] for d in range(2)))

    for actual, expected in zip(
        errors + outputs + state,
        example.errors + example.outputs + example.final_state,
        strict=True,
    ):
        assert isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
            actual,
            expected,
        )
    print("Verified: sequential and regrouped errors, outputs, and final state agree.")
    print(f"Errors: {example.errors}; outputs: {example.outputs}")


if __name__ == "__main__":
    verify_example()
