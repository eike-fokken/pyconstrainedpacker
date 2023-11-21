from pyconstrainedpacker import ArticleShipment, PackingGroup, allocate


def test_ArticleShipment_construction() -> None:
    shipment_dict = {250.0: 8, 500.0: 20}

    ArticleShipment(shipment_dict)
    PackingGroup("first", 4000, 0.8, 2)


def test_simple_run() -> None:
    shipment_dict = {
        250.0: 8,
        500.0: 20,
    }
    group_names = ["One", "Two", "Three"]
    group_demands = [4832.0, 5923.0, 6981.0]

    # minimal_allocation_factor = 1.0
    minimal_allocation_factor = 0.9
    package_sizes = list(shipment_dict.keys())
    packages_shipped_per_size = list(shipment_dict.values())

    allocate(
        group_names,
        group_demands,
        minimal_allocation_factor,
        package_sizes,
        packages_shipped_per_size,
    )
