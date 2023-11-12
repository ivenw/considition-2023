from pathlib import Path

from src.data_loading import Location, load_general_data, load_map_data
from src.scoring import (
    score,
    get_distance_matrix,
    get_f3100_count_vector,
    get_f9100_count_vector,
    get_sales_volume_vector,
    get_footfall_vector,
    combine_refill_station_count_vectors,
    score_vectorized,
)


def naive_algo(locations: list[Location]) -> list[Location]:
    result = []
    for location in locations:
        if location.sales_volume > 100:
            location.f3100_count = 1
            location.f9100_count = 0
        result.append(location)

    return result


def main():
    locations = load_map_data(Path("data/vasteras.json"))
    general_data = load_general_data(Path("data/general.json"))
    solution = naive_algo(locations)

    distance_matrix = get_distance_matrix(locations)
    sales_volume_vector = get_sales_volume_vector(locations)
    footfall_vector = get_footfall_vector(locations)

    f3100_count_vector = get_f3100_count_vector(locations)
    f9100_count_vector = get_f9100_count_vector(locations)
    refill_station_count_vector = combine_refill_station_count_vectors(
        f3100_count_vector, f9100_count_vector
    )

    score_vectorized(
        general_data,
        distance_matrix,
        sales_volume_vector,
        footfall_vector,
        refill_station_count_vector,
    )

    total_score = 0
    for location in solution:
        total_score += score(location, general_data)

    print(f"Total score: {total_score}")


if __name__ == "__main__":
    main()
