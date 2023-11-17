from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Any

import numpy as np
import pygad
from numpy.typing import NDArray

from src.data_loading import Location, load_general_data, load_map_data
from src.scoring import (
    ScoringData,
    score_vectorized,
)


NUM_GENERATIONS = 1_000
SOL_PER_POP = 1_000
FITNESS_BATCH_SIZE = SOL_PER_POP

NUM_PARENTS_MATING = 4
PARENT_SELECTION_TYPE = "sss"
KEEP_PARENTS = 1

CROSSOVER_TYPE = "single_point"

MUTATION_TYPE = "random"
MUTATION_PERCENT_GENES = 10

GENE_TO_LOCATION_MAP = np.array(
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 4),
        (0, 5),
    ]
)


def naive_algo(locations: list[Location]) -> NDArray[np.uint32]:
    # should score 429409.0
    result = []
    for location in locations:
        if location.sales_volume > 100:
            location.f3100_count = 1
            location.f9100_count = 3
        result.append([location.f3100_count, location.f9100_count])

    return np.array([result, result], dtype=np.uint32)


def run_performance_test(function: Callable[[], Any], *args, **kwargs):
    runs = []
    for _ in range(100_000):
        start = perf_counter_ns()
        function(*args, **kwargs)
        stop = perf_counter_ns()
        runs.append(stop - start)

    print(f"Average time: {sum(runs) / len(runs)} ns")


def on_generation(ga_instance: pygad.GA):
    if ga_instance.generations_completed % 50 == 0:
        print(
            f"Generation {ga_instance.generations_completed}\n",
            f"best solution score: {ga_instance.best_solution()[1]}",
        )


def main():
    locations = load_map_data(Path("data/vasteras.json"))
    general_data = load_general_data(Path("data/general.json"))
    scoring_data = ScoringData.create(locations, general_data)
    num_genes = len(locations)

    def fitness_func(
        ga_instance: pygad.GA,
        solutions: NDArray[np.int32],
        solution_idx: NDArray[np.int32],
    ):
        del ga_instance
        del solution_idx

        mapped_solutions = GENE_TO_LOCATION_MAP[solutions]

        result = score_vectorized(
            general_data,
            scoring_data,
            mapped_solutions,
        )

        return result

    ga_instance = pygad.GA(
        fitness_func=fitness_func,
        on_generation=on_generation,
        num_generations=NUM_GENERATIONS,
        sol_per_pop=SOL_PER_POP,
        fitness_batch_size=FITNESS_BATCH_SIZE,
        num_parents_mating=NUM_PARENTS_MATING,
        parent_selection_type=PARENT_SELECTION_TYPE,
        # keep_parents=KEEP_PARENTS,
        keep_parents=2,
        crossover_type=CROSSOVER_TYPE,
        mutation_type=MUTATION_TYPE,
        mutation_percent_genes=MUTATION_PERCENT_GENES,
        num_genes=num_genes,
        gene_type=np.int32,  # type: ignore
        gene_space=np.arange(0, 21, dtype=np.int32),
    )
    ga_instance.run()

    print(ga_instance.initial_population)
    print(ga_instance.population)
    print(ga_instance.best_solution())

    total_score = score_vectorized(
        general_data,
        scoring_data,
        ga_instance.best_solution()[0],
    )
    print(f"Total score, vectorized: {total_score}")


if __name__ == "__main__":
    main()
