from pyconstrainedpacker.MockDemand import create_groups, make_demands_from_groups


def test_create_groups() -> None:

    total_head_count = 5000
    default_group_size = 50

    groups, counts = create_groups(total_head_count, default_group_size)

    for i in range(len(groups)):
        print(f"{groups[i]}: {counts[i]}")

    assert sum(counts) == total_head_count


def test_make_demands_from_groups() -> None:

    total_head_count = 5000
    default_group_size = 50
    demand_per_person = 80
    group_names, head_counts = create_groups(total_head_count, default_group_size)

    group_names, demands = make_demands_from_groups(
        group_names, head_counts, demand_per_person
    )

    for i in range(len(group_names)):
        print(f"{group_names[i]}: {demands[i]}")

    assert sum(demands) == total_head_count * demand_per_person
    print(f"total_demand: {total_head_count * demand_per_person}")
