from pathlib import Path
import json

from dataclasses import dataclass
from data_keys import LocationKeys, CoordinateKeys, GeneralKeys


@dataclass
class Coordinate:
    lat: float
    long: float


@dataclass
class Location:
    f3100_count: int
    f9100_count: int
    sales_volume: float
    footfall: float
    coordinate: Coordinate


@dataclass
class GeneralData:
    co2_price: float
    f3100_co2: float
    f9100_co2: float
    f3100_refill_capacity: int
    f9100_refill_capacity: int
    f3100_leasing_cost: float
    f9100_leasing_cost: float
    classic_co2_per_unit: float
    refill_co2_per_unit: float
    refill_profit_per_unit: float
    refill_sales_factor: float
    wiling_to_travel: float
    exp_distribution_factor: float


def load_map_data(json_file: Path) -> list[Location]:
    result = []

    with open(json_file, "r", encoding="utf8") as f:
        map_data = json.load(f)

    for location in map_data[LocationKeys.locations].values():
        location = Location(
            f3100_count=0,
            f9100_count=0,
            sales_volume=location[LocationKeys.salesVolume],
            footfall=location[LocationKeys.footfall],
            coordinate=Coordinate(
                lat=location[CoordinateKeys.latitude],
                long=location[CoordinateKeys.longitude],
            ),
        )
        result.append(location)

    return result


def load_general_data(json_file: Path) -> GeneralData:
    with open(json_file, "r", encoding="utf8") as f:
        general_data = json.load(f)

    return GeneralData(
        co2_price=general_data[GeneralKeys.co2PricePerKiloInSek],
        f3100_co2=general_data[GeneralKeys.f3100Data][GeneralKeys.staticCo2],
        f9100_co2=general_data[GeneralKeys.f9100Data][GeneralKeys.staticCo2],
        f3100_refill_capacity=general_data[GeneralKeys.f3100Data][
            GeneralKeys.refillCapacityPerWeek
        ],
        f9100_refill_capacity=general_data[GeneralKeys.f9100Data][
            GeneralKeys.refillCapacityPerWeek
        ],
        f3100_leasing_cost=general_data[GeneralKeys.f3100Data][
            GeneralKeys.leasingCostPerWeek
        ],
        f9100_leasing_cost=general_data[GeneralKeys.f9100Data][
            GeneralKeys.leasingCostPerWeek
        ],
        classic_co2_per_unit=general_data[GeneralKeys.classicUnitData][
            GeneralKeys.co2PerUnitInGrams
        ],
        refill_co2_per_unit=general_data[GeneralKeys.refillUnitData][
            GeneralKeys.co2PerUnitInGrams
        ],
        refill_profit_per_unit=general_data[GeneralKeys.refillUnitData][
            GeneralKeys.profitPerUnit
        ],
        refill_sales_factor=general_data[GeneralKeys.refillSalesFactor],
        wiling_to_travel=general_data[GeneralKeys.willingnessToTravelInMeters],
        exp_distribution_factor=general_data[
            GeneralKeys.constantExpDistributionFunction
        ],
    )
