import math
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from src.data_loading import Coordinate, GeneralData, Location


@dataclass
class ScoringData:
    distance_matrix: NDArray[np.float32]
    sales_volume_vector: NDArray[np.float32]
    sales_capacity_vector: NDArray[np.float32]
    leasing_cost_vector: NDArray[np.float32]
    co2_produced_vector: NDArray[np.float32]
    footfall_vector: NDArray[np.float32]

    @classmethod
    def create(cls, locations: list[Location], general_data: GeneralData) -> Self:
        return cls(
            distance_matrix=create_distance_matrix(locations),
            sales_volume_vector=create_sales_volume_vector(locations, general_data),
            sales_capacity_vector=create_sales_capacity_vector(general_data),
            leasing_cost_vector=create_leasting_cost_vector(general_data),
            co2_produced_vector=create_co2_produced_vector(general_data),
            footfall_vector=create_footfall_vector(locations),
        )


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


def create_distance_matrix(locations: list[Location]) -> NDArray[np.float32]:
    result = np.zeros((len(locations), len(locations)), dtype=np.float32)

    for i in range(len(locations)):
        for j in range(len(locations)):
            if i == j:
                result[i, j] = 0
                continue

            result[i, j] = distance(locations[i].coordinate, locations[j].coordinate)

    return result


def create_sales_volume_vector(
    locations: list[Location], general_data: GeneralData
) -> NDArray[np.float32]:
    result = np.zeros(len(locations), dtype=np.float32)
    for i in range(len(locations)):
        result[i] = locations[i].sales_volume * general_data.refill_sales_factor
    return result


def create_sales_capacity_vector(general_data: GeneralData) -> NDArray[np.float32]:
    return np.array(
        [general_data.f3100_refill_capacity, general_data.f9100_refill_capacity],
        dtype=np.float32,
    )


def create_leasting_cost_vector(general_data: GeneralData) -> NDArray[np.float32]:
    return np.array(
        [general_data.f3100_leasing_cost, general_data.f9100_leasing_cost],
        dtype=np.float32,
    )


def create_co2_produced_vector(general_data: GeneralData) -> NDArray[np.float32]:
    return np.array([general_data.f3100_co2, general_data.f9100_co2], dtype=np.float32)


def create_footfall_vector(locations: list[Location]) -> NDArray[np.float32]:
    result = np.zeros(len(locations), dtype=np.float32)
    for i in range(len(locations)):
        result[i] = locations[i].footfall
    return result


def location_has_refill_station(
    solution: NDArray[np.uint32],
) -> NDArray[np.bool_]:
    """Get a mask for the distance matrix based on the refill station vector.

    The mask is a boolean matrix where True means that the location has
    a refill station.
    """
    result = np.any(solution, axis=2)
    return result


def score_vectorized(
    general_data: GeneralData,
    scoring_data: ScoringData,
    solution: NDArray[np.uint32],
):
    # n = number of solutions
    # m = number of locations

    has_refill_station = location_has_refill_station(solution)  # (n, m)

    distances = np.where(
        scoring_data.distance_matrix < general_data.wiling_to_travel,
        scoring_data.distance_matrix,
        0,
    )
    distances = np.where(distances == 0, np.nan, distances)  # (m, m)

    distribution_factors = np.nan_to_num(
        np.power(
            general_data.exp_distribution_factor,
            general_data.wiling_to_travel - distances,
        )
        - 1
    )  # (m, m)

    distribution_factors = distribution_factors[np.newaxis, :, :].repeat(
        len(solution), axis=0
    )

    filtered_distribution_factors = np.zeros_like(distribution_factors)
    filtered_distribution_factors[~has_refill_station] = distribution_factors[
        ~has_refill_station
    ]

    total_distribution_factor = filtered_distribution_factors.sum(axis=1)[:, np.newaxis]

    # i think here it's faling apart
    redistributed_sales = np.nan_to_num(
        np.divide(
            filtered_distribution_factors,
            total_distribution_factor,
            out=np.zeros_like(filtered_distribution_factors),
            where=total_distribution_factor != 0,
        )
        * general_data.refill_distribution_rate
        * scoring_data.sales_volume_vector[np.newaxis, :].T
    ).sum(axis=1)

    sales_volume = scoring_data.sales_volume_vector[np.newaxis, :].repeat(
        len(solution), axis=0
    )
    sales_volume = np.where(has_refill_station, sales_volume, 0)
    sales_volume = redistributed_sales + sales_volume

    sales_capacity = np.sum(solution * scoring_data.sales_capacity_vector, axis=2)

    sales_volume = sales_volume.round(0).clip(0, sales_capacity)

    total_revenue = (sales_volume * general_data.refill_profit_per_unit).sum(axis=1)

    total_leasing_cost = (
        (solution * scoring_data.leasing_cost_vector).sum(axis=2).sum(axis=1)
    )

    total_earnings = total_revenue - total_leasing_cost

    total_co2_produced = (
        (solution * scoring_data.co2_produced_vector / 1_000)
        .sum(axis=2)
        .sum(axis=1)
        .round(0)
    )

    total_co2_savings = (
        (
            sales_volume
            * (general_data.classic_co2_per_unit - general_data.refill_co2_per_unit)
            / 1_000
        )
        .sum(axis=1)
        .round()
    )

    total_co2 = total_co2_savings - total_co2_produced

    total_footfall = np.where(
        has_refill_station,
        # the repeat can be moved out into the scoring data
        scoring_data.footfall_vector[np.newaxis, :].repeat(len(solution), axis=0),
        0,
    ).sum(axis=1)

    score = (total_co2 * general_data.co2_price + total_earnings) * (1 + total_footfall)

    return score.round(0)
