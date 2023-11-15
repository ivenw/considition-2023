import numpy as np
from numpy.typing import NDArray

LOCATION_MAP = np.array(
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


# TODO assigning lambda functions to variables is not idiomatic
mapper_func = lambda x: (LOCATION_MAP[x][0], LOCATION_MAP[x][1])
# TODO under the hood this is just a for loop, so there might be some peformance on the table
vmapper_func = np.vectorize(mapper_func)


def map_solutions(solutions: NDArray[np.int32]) -> NDArray[np.int32]:
    solutions_flattened = solutions.flatten()
    mapped_array = vmapper_func(solutions_flattened)
    mapped_array = np.insert(
        mapped_array[1], np.arange(len(mapped_array[0])), mapped_array[0]
    )
    mapped_solutions = mapped_array.reshape(solutions.shape[0], solutions.shape[1] * 2)
    return mapped_solutions

