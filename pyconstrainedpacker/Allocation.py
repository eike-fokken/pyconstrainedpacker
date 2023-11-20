from collections import OrderedDict
from typing import List

import casadi
import numpy as np

from .ArticleShipment import ArticleShipment
from .PackingGroup import PackingGroup


def allocate(
    group_names: List[str],
    group_demands: List[float],
    minimal_allocation_factor: float,
    package_sizes: List[float],
    packages_shipped_per_size: List[int],
) -> None:
    check_allocate_inputs(
        group_names,
        group_demands,
        minimal_allocation_factor,
        package_sizes,
        packages_shipped_per_size,
    )

    total_demand = sum(group_demands)
    total_supply = sum(
        [
            size * number
            for size, number in zip(package_sizes, packages_shipped_per_size)
        ]
    )

    demand_reduction_factor = min(1, total_supply / total_demand)

    demand_by_group = dict(zip(group_names, group_demands))
    reduced_demand_by_group = dict(
        (key, demand_reduction_factor * value) for key, value in demand_by_group.items()
    )

    shipped_packages_by_size = dict(zip(package_sizes, packages_shipped_per_size))
    groups = [
        PackingGroup(
            name,
            reduced_demand_by_group[name],
            minimal_allocation_factor,
            len(package_sizes),
        )
        for name in reduced_demand_by_group
    ]

    shipment = ArticleShipment(shipped_packages_by_size)
    variable_data = [group.declare_variables_and_their_bounds() for group in groups]
    variable_lower_bounds = np.hstack([data[0] for data in variable_data])
    variables = casadi.vertcat(*[data[1] for data in variable_data])
    variable_upper_bounds = np.hstack([data[2] for data in variable_data])
    discrete_list_of_lists = [data[3] for data in variable_data]
    discrete = [item for sublist in discrete_list_of_lists for item in sublist]

    package_sizes_nparray = np.array(package_sizes)
    group_constraint_data = [
        group.set_constraints(package_sizes_nparray) for group in groups
    ]
    constraint_lower_bounds_list = [data[0] for data in group_constraint_data]
    constraint_list = [data[1] for data in group_constraint_data]
    constraint_upper_bounds_list = [data[2] for data in group_constraint_data]

    shipment_constraint_data = shipment.set_constraints(groups)
    constraint_lower_bounds_list.append(shipment_constraint_data[0])
    constraint_list.append(shipment_constraint_data[1])
    constraint_upper_bounds_list.append(shipment_constraint_data[2])

    # for item in constraint_list:
    #     print(item)
    #     print("\n")

    # print(casadi.vertcat(*constraint_list))

    # raise ValueError("AAAA")

    constraint_lower_bounds = np.hstack(constraint_lower_bounds_list)

    print(constraint_lower_bounds)

    constraints = casadi.vertcat(*constraint_list)
    constraint_upper_bounds = np.hstack(constraint_upper_bounds_list)

    objective = sum([group.set_objective() for group in groups])

    problem = {"x": variables, "g": constraints, "f": objective}
    solver = casadi.qpsol("solver", "highs", problem, {"discrete": discrete})

    x0 = np.zeros(len(groups) * (len(package_sizes) + 2))

    print(f"{x0=}")
    print(f"{variable_lower_bounds=}")
    print(f"{variable_upper_bounds=}")
    print(f"{constraint_lower_bounds=}")
    print(f"{constraint_upper_bounds=}")

    solver(
        x0=x0,
        lbx=variable_lower_bounds,
        ubx=variable_upper_bounds,
        lbg=constraint_lower_bounds,
        ubg=constraint_upper_bounds,
    )


def check_allocate_inputs(
    group_names: List[str],
    group_demands: List[float],
    minimal_allocation_factor: float,
    package_sizes: List[float],
    packages_shipped_per_size: List[int],
) -> None:
    multiply_defined_names_raw = [
        name for name in group_names if group_names.count(name) > 1
    ]
    if len(multiply_defined_names_raw) > 0:
        multiply_defined_names = list(OrderedDict.fromkeys(multiply_defined_names_raw))
        error_message = (
            "Group names are not unique!"
            + "Multiply defined group names are:"
            + "\n".join(multiply_defined_names)
        )
        raise ValueError(error_message)
    if len(group_demands) != len(group_names):
        raise ValueError("Mismatch in number of group names and group demands!")

    demand_by_group = dict(zip(group_names, group_demands))

    negative_demand_groups = [
        name for name, demand in demand_by_group.items() if demand < 0
    ]
    if len(negative_demand_groups) > 0:
        error_message = "The following groups have negative demand:" + "\n".join(
            negative_demand_groups
        )

    if minimal_allocation_factor < 0 or minimal_allocation_factor > 1:
        raise ValueError(
            "Minimal allocation factor must"
            + f" lie between 0 and one, but got: {minimal_allocation_factor}"
        )

    if len(package_sizes) != len(packages_shipped_per_size):
        raise ValueError(
            "Must have a number of packages shipped for each package size!"
        )
    negative_package_sizes = [size for size in package_sizes if size <= 0]
    if len(negative_package_sizes) > 0:
        raise ValueError("Can't have negative package size!")

    if package_sizes != sorted(package_sizes):
        raise ValueError("package sizes must be sorted from smallest to greatest!")

    negative_packages_shipped = [
        number for number in packages_shipped_per_size if number <= 0
    ]
    if len(negative_packages_shipped) > 0:
        raise ValueError("Can't have negative number of packages shipped!")
