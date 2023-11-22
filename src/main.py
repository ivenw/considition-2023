import os
from pathlib import Path
from time import perf_counter_ns
import time
from typing import Any, Callable
import concurrent.futures
import multiprocessing as mp

import numpy as np
import pygad
from numpy.typing import NDArray

from src.api import getGeneralData, getMapData, submit
from src.data_keys import (
    LocationKeys as LK,
)
from src.data_keys import (
    MapNames as MN,
)
from src.data_keys import (
    ScoringKeys,
)
from src.data_keys import (
    ScoringKeys as SK,
)
from src.data_loading import Location, load_general_data, load_map_data
from src.scoring import (
    ScoringData,
    score_vectorized,
)
from starter_kit.scoring import calculateScore

UPLOAD = False
MAX_PROCESSES = int(mp.cpu_count()/2)

PRELOAD = True # set to false to run GA from scratch


NUM_GENERATIONS = 50
SOL_PER_POP = 500  # Should always be divisible by 100
assert SOL_PER_POP % 100 == 0, "SOL_PER_POP must be divisible by 100"
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
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
)

map_name = "goteborg"

locations, map_name = load_map_data(Path(f"data/{map_name}.json"))
general_data = load_general_data(Path("data/general.json"))
scoring_data = ScoringData.create(locations, general_data)
num_genes = len(locations)


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


def run_ga_instance(ga_instance: pygad.GA):
    ga_instance.run()
    return ga_instance


def save_best_pop_and_sol(map_name: str, best_ga_instance: pygad.GA):
    best_solution = best_ga_instance.best_solution()

    if not Path("./best_populations/").exists():
        os.mkdir(Path("./best_populations/"))
    if not Path("./best_solutions/").exists():
        os.mkdir(Path("./best_solutions/"))

    print("Saving best population...")
    np.save(
        os.path.abspath(
            Path(
                f"best_populations/{map_name}_{best_ga_instance.population.shape[0]}_{int(best_solution[1])}.npy"
            )
        ),
        best_ga_instance.population,
    )
    print("Saving best solution...")
    np.save(
        os.path.abspath(
            Path(
                f"best_solutions/{map_name}_{time.time_ns()}_{int(best_solution[1])}.npy"
            )
        ),
        best_solution[0],
    )


def preload_population(map_name: str) -> tuple[np.ndarray, int]:
    if not os.path.exists(Path("./best_populations/")):
        print("No best populations found")
        return None

    best_populations = [
        file
        for file in os.listdir(Path("./best_populations/"))
        if file.startswith(map_name)
    ]

    if not best_populations:
        print("No best populations found")
        return None

    best_populations.sort(
        key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True
    )
    best_population = best_populations[0]
    print(f"Loading best population {best_population}")
    best_population = np.load(Path(f"./best_populations/{best_population}"))

    print(best_population)
    if best_population.shape[0] != SOL_PER_POP:
        print(
            f"Best population shape {best_population.shape[0]} does not match SOL_PER_POP {SOL_PER_POP}, repeating best population"
        )
        best_population = np.repeat(
            best_population, SOL_PER_POP / best_population.shape[0], axis=0
        )

    return best_population


def main():
    if PRELOAD:
        best_population = preload_population(map_name)
        if best_population is not None:
            ga_instances = [
                pygad.GA(
                    fitness_func=fitness_func,
                    on_generation=on_generation,
                    num_generations=NUM_GENERATIONS,
                    sol_per_pop=SOL_PER_POP,
                    fitness_batch_size=FITNESS_BATCH_SIZE,
                    num_parents_mating=NUM_PARENTS_MATING,
                    parent_selection_type=PARENT_SELECTION_TYPE,
                    keep_elitism=KEEP_PARENTS,
                    crossover_type=CROSSOVER_TYPE,
                    mutation_type=MUTATION_TYPE,
                    mutation_percent_genes=MUTATION_PERCENT_GENES,
                    num_genes=num_genes,
                    initial_population=best_population,
                    gene_type=np.int32,  # type: ignore
                    gene_space=np.arange(0, 9, dtype=np.int32),
                )
                for _ in range(MAX_PROCESSES)
            ]
    else:
        # Create number of instances of the GA class
        ga_instances = [
            pygad.GA(
                fitness_func=fitness_func,
                on_generation=on_generation,
                num_generations=NUM_GENERATIONS,
                sol_per_pop=SOL_PER_POP,
                fitness_batch_size=FITNESS_BATCH_SIZE,
                num_parents_mating=NUM_PARENTS_MATING,
                parent_selection_type=PARENT_SELECTION_TYPE,
                keep_elitism=KEEP_PARENTS,
                crossover_type=CROSSOVER_TYPE,
                mutation_type=MUTATION_TYPE,
                mutation_percent_genes=MUTATION_PERCENT_GENES,
                num_genes=num_genes,
                gene_type=np.int32,  # type: ignore
                gene_space=np.arange(0, 9, dtype=np.int32),
            )
            for _ in range(MAX_PROCESSES)
        ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        ga_instances = [
            executor.submit(run_ga_instance, ga_instance)
            for ga_instance in ga_instances
        ]

        best_score = 0
        for future in concurrent.futures.as_completed(ga_instances):
            ga_instance = future.result()
            # Get the best solution from all runs
            current_score = ga_instance.best_solution()[1]
            if best_score < current_score:
                best_score = current_score
                best_ga_instance = ga_instance

    save_best_pop_and_sol(map_name, best_ga_instance)

    api_key = os.environ["apiKey"]
    solution_array = GENE_TO_LOCATION_MAP[best_ga_instance.best_solution()[0]]
    mapEntity = getMapData(map_name, api_key)
    generalData = getGeneralData()

    solution = {LK.locations: {}}
    for key, a in zip(mapEntity[LK.locations], solution_array):
        location = mapEntity[LK.locations][key]
        name = location[LK.locationName]

        solution[LK.locations][name] = {
            LK.f3100Count: int(a[0]),
            LK.f9100Count: int(a[1]),
        }

    # filter solutions for any locations that don't contain any f3100 or f9100
    solution[LK.locations] = {
        key: value
        for key, value in solution[LK.locations].items()
        if value[LK.f3100Count] > 0 or value[LK.f9100Count] > 0
    }

    score = calculateScore(map_name, solution, mapEntity, generalData)

    id_ = score[SK.gameId]
    print(f"Storing  game with id {id_}.")
    print(f"Score: {score[SK.gameScore][SK.total]}")
    print(f"CO2: {score[SK.gameScore][SK.co2Savings]}")
    print(f"Footfall: {score[SK.gameScore][SK.totalFootfall]}")
    print(f"Revenue: {score[SK.totalRevenue]}")

    if not UPLOAD:
        return
    scored_solution = submit(map_name, solution, api_key)
    if scored_solution:
        print("Successfully submitted game")
        print(f"id: {scored_solution[ScoringKeys.gameId]}")
        print(f"Score: {scored_solution[ScoringKeys.gameScore]}")


if __name__ == "__main__":
    main()
