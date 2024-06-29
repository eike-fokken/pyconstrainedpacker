from pyconstrainedpacker import ArticleShipment, PackingGroup, allocate
from pyconstrainedpacker.MockDemand import create_groups, make_demands_from_groups


def test_simple_run() -> None:

    total_head_count = 5000
    default_group_size = 50
    demand_per_person = 80
    group_names, head_counts = create_groups(
        total_head_count, default_group_size, random_seed=12
    )

    group_names, demands = make_demands_from_groups(
        group_names, head_counts, demand_per_person
    )

    total_demand = total_head_count * demand_per_person
    print(f"{total_demand=}")
    shipment_dict = {
        250.0: 400,
        400.0: 500,
    }

    total_supply = sum([size * value for size, value in shipment_dict.items()])
    print(f"{total_supply=}")

    minimal_allocation_factor = 0.5
    package_sizes = list(shipment_dict.keys())
    packages_shipped_per_size = list(shipment_dict.values())

    allocate(
        group_names,
        demands,
        minimal_allocation_factor,
        package_sizes,
        packages_shipped_per_size,
    )
