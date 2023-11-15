import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Any

import numpy as np
import pygad
from numpy.typing import NDArray

from src.data_loading import Location, load_general_data, load_map_data
from src.scoring import (
    combine_refill_station_count_vectors,
    get_distance_matrix,
    get_f3100_count_vector,
    get_f9100_count_vector,
    get_footfall_vector,
    get_sales_volume_vector,
    score,
    score_vectorized,
)
from src.utils import map_solutions

NUM_GENERATIONS = 100
NUM_PARENTS_MATING = 4
SOL_PER_POP = 15
FITNESS_BATCH_SIZE = SOL_PER_POP


def naive_algo(locations: list[Location]) -> list[Location]:
    result = []
    for location in locations:
        if location.sales_volume > 100:
            location.f3100_count = 1
            location.f9100_count = 0
        result.append(location)

    return result


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
    solution = naive_algo(locations)

    num_genes = len(locations)

    def fitness_func(ga_instance: pygad.GA, solutions: NDArray[np.int32], solution_idx):
        mapped_solutions = map_solutions(solutions)
        # TODO: Add vectorized scoring function
        return mapped_solutions

    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=SOL_PER_POP,
        fitness_batch_size=FITNESS_BATCH_SIZE,
        num_genes=num_genes,
        on_generation=on_generation,
        gene_type=np.int32,  # type: ignore
        gene_space=np.linspace(0, 20, 21, dtype=np.int32),
    )
    ga_instance.run()

    print(ga_instance.initial_population)
    print(ga_instance.population)
    print(ga_instance.best_solution())

    sys.exit(0)

    distance_matrix = get_distance_matrix(locations)
    sales_volume_vector = get_sales_volume_vector(locations)
    footfall_vector = get_footfall_vector(locations)

    f3100_count_vector = get_f3100_count_vector(solution)
    f9100_count_vector = get_f9100_count_vector(solution)
    refill_station_count_vector = combine_refill_station_count_vectors(
        f3100_count_vector, f9100_count_vector
    )

    """print ("Running performance test for score_vectorized")
    run_performance_test(
        score_vectorized,
        general_data,
        distance_matrix,
        sales_volume_vector,
        footfall_vector,
        refill_station_count_vector,
    )"""

    """total_score = score_vectorized(
        general_data,
        distance_matrix,
        sales_volume_vector,
        footfall_vector,
        refill_station_count_vector,
    )
    print(f"Total score, vectorized: {total_score}")"""

    """print ("Running performance test for naive_score")
    run_performance_test(naive_score, solution, general_data)"""
    total_score = vanilla_score(solution, general_data)
    print(f"Total score: {total_score}")


if __name__ == "__main__":
    main()
