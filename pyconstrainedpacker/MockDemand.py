from math import floor

import numpy as np


def make_demands_from_groups(
    group_names: list[str], head_counts: list[int], demand_per_person: float
) -> tuple[list[str], list[float]]:
    """Create a demand from a headcount list."""

    return group_names, [count * demand_per_person for count in head_counts]


def create_groups(
    total_head_count: int, default_group_size: int = 50, random_seed: int | None = None
) -> tuple[list[str], list[int]]:

    remaining_people = total_head_count
    # MAV contains about a 12th of all people.
    MAV_headcount = int(floor(remaining_people / 12))
    if MAV_headcount < default_group_size:
        MAV_headcount = default_group_size

    remaining_people -= MAV_headcount

    group_names = ["MAV"]
    head_counts = [MAV_headcount]

    myrng = np.random.default_rng(seed=random_seed)

    while remaining_people > 0:
        group_size_approximation = myrng.normal(
            default_group_size, 0.25 * default_group_size, 1
        )[0]

        # Cut off too small values:
        group_size_approximation = max(
            group_size_approximation, 0.25 * default_group_size
        )
        # Cut off too big values:
        group_size_approximation = min(
            group_size_approximation, 1.75 * default_group_size
        )

        # make an integer:
        group_size = int(floor(group_size_approximation))

        if group_size > remaining_people:
            group_size = remaining_people

        remaining_people -= group_size
        group_names.append(f"group_{len(group_names)}")
        head_counts.append(group_size)

    return group_names, head_counts
