import math

import numpy as np
from numpy.typing import NDArray

from src.data_loading import Coordinate, GeneralData, Location


def distance(c_a: Coordinate, c_b: Coordinate) -> float:
    R = 6371e3
    φ_a = c_a.lat * math.pi / 180  #  φ, λ in radians
    φ_b = c_b.lat * math.pi / 180
    Δφ = (c_b.lat - c_a.lat) * math.pi / 180
    Δλ = (c_b.long - c_a.long) * math.pi / 180
    a = math.sin(Δφ / 2) * math.sin(Δφ / 2) + math.cos(φ_a) * math.cos(φ_b) * math.sin(
        Δλ / 2
    ) * math.sin(Δλ / 2)
    d = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return round(d, 0)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def get_f3100_count_vector(locations: list[Location]) -> NDArray[np.uint32]:
    result = np.zeros(len(locations), dtype=np.uint32)
    for i in range(len(locations)):
        result[i] = locations[i].f3100_count
    return result


def get_f9100_count_vector(locations: list[Location]) -> NDArray[np.uint32]:
    result = np.zeros(len(locations), dtype=np.uint32)
    for i in range(len(locations)):
        result[i] = locations[i].f9100_count
    return result


def get_sales_volume_vector(locations: list[Location]) -> NDArray[np.float32]:
    result = np.zeros(len(locations), dtype=np.float32)
    for i in range(len(locations)):
        result[i] = locations[i].sales_volume
    return result


def get_footfall_vector(locations: list[Location]) -> NDArray[np.float32]:
    result = np.zeros(len(locations), dtype=np.float32)
    for i in range(len(locations)):
        result[i] = locations[i].footfall
    return result


def get_distance_matrix(locations: list[Location]) -> NDArray[np.float32]:
    result = np.zeros((len(locations), len(locations)), dtype=np.float32)

    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j:
                result[i, j] = 0
                continue

            result[i, j] = distance(locations[i].coordinate, locations[j].coordinate)

    return result


def combine_refill_station_count_vectors(
    a: NDArray[np.uint32], b: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    """Combine two vectors of refill station counts into a single vector.

    F3100 is in the first column, F9100 is in the second column.
    """
    return np.hstack((a.reshape(-1, 1), b.reshape(-1, 1)))


def location_has_refill_station(
    refill_station_counts: NDArray[np.uint32],
) -> NDArray[np.bool_]:
    """Get a mask for the distance matrix based on the refill station vector.

    The mask is a boolean matrix where True means that the location has
    a refill station.
    """
    return np.any(refill_station_counts, axis=1)


def redistribute_sales(
    general_data: GeneralData,
    distance_matrix: NDArray[np.float32],
    sales_volume_vector: NDArray[np.float32],
    refill_station_counts: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Redistribute sales from locations without refill stations to locations with
    refill stations.

    The sales are redistributed based on the distance matrix and the refill station
    counts.
    """
    distance_matrix_refill_filtered = distance_matrix * location_has_refill_station(
        refill_station_counts
    )
    distance_matrix_willingness_filtered = np.where(
        distance_matrix_refill_filtered < general_data.wiling_to_travel,
        distance_matrix_refill_filtered,
        0,
    )
    distribution_factor_matrix = (
        distance_matrix_willingness_filtered**general_data.exp_distribution_factor
    )

    # not sure that this is correct
    return np.round(
        np.sum(distribution_factor_matrix * sales_volume_vector.reshape(-1, 1), axis=0),
        0,
    )


def score_vectorized(
    general_data: GeneralData,
    distance_matrix: NDArray[np.float32],
    sales_volume_vector: NDArray[np.float32],
    footfall_vector: NDArray[np.float32],
    refill_station_counts: NDArray[np.uint32],
):
    sales_volume = redistribute_sales(
        general_data, distance_matrix, sales_volume_vector, refill_station_counts
    )

    sales_volume = np.round(sales_volume * general_data.refill_sales_factor, 0)

    # not sure this is correct
    sales_capacity = np.sum(
        refill_station_counts
        * [general_data.f3100_refill_capacity, general_data.f9100_refill_capacity],
        axis=1,
    )

    sales_volume = np.clip(sales_volume, 0, sales_capacity)

    # not sure this is correct
    leasing_cost = np.sum(
        refill_station_counts
        * [general_data.f3100_leasing_cost, general_data.f9100_leasing_cost],
        axis=1,
    )

    revenue = sales_volume * general_data.refill_profit_per_unit
    earnings = revenue - leasing_cost

    # not sure this is correct
    co2_procuced = np.sum(
        refill_station_counts * [general_data.f3100_co2, general_data.f9100_co2],
        axis=1,
    )

    co2_savings = (
        sales_volume
        * (general_data.classic_co2_per_unit - general_data.refill_co2_per_unit)
        / 1_000
    )

    co2_savings_adjusted = co2_savings - co2_procuced

    score = (co2_savings_adjusted * general_data.co2_price + earnings) * (
        1 + footfall_vector
    )

    return np.sum(score)


def score(location: Location, general_data: GeneralData) -> float:
    sales_volume = location.sales_volume * general_data.refill_sales_factor
    sales_capacity = (
        location.f3100_count * general_data.f3100_refill_capacity
        + location.f9100_count * general_data.f9100_refill_capacity
    )
    sales_volume = round(sales_volume, 0)
    sales_volume = clamp(sales_volume, 0, sales_capacity)
    leasing_cost = (
        location.f3100_count * general_data.f3100_leasing_cost
        + location.f9100_count * general_data.f9100_leasing_cost
    )

    revenue = sales_volume * general_data.refill_profit_per_unit
    earnings = revenue - leasing_cost

    f3100_co2_produced = location.f3100_count * general_data.f3100_co2 / 1_000
    f9100_co2_produced = location.f9100_count * general_data.f9100_co2 / 1_000

    co2_savings = (
        sales_volume
        * (general_data.classic_co2_per_unit - general_data.refill_co2_per_unit)
        / 1_000
    )

    co2_savings_adjusted = co2_savings - f3100_co2_produced - f9100_co2_produced

    score = (co2_savings_adjusted * general_data.co2_price + earnings) * (
        1 + location.footfall
    )

    return score
