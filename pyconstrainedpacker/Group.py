from typing import List, Tuple

import casadi
import numpy as np
import numpy.typing as npt


class Group:
    def __init__(
        self,
        name: str,
        demand: float,
        minimal_allocation_factor: float,
        number_of_package_sizes: int,
    ) -> None:
        assert demand >= 0
        assert 0 <= minimal_allocation_factor <= 1
        assert number_of_package_sizes >= 1

        self.name = name
        self.demand = demand
        self.minimal_allocation_factor = minimal_allocation_factor
        self.allocations: casadi.MX = casadi.MX.sym(
            "allocations_" + name, number_of_package_sizes
        )
        self.positive_deviation = casadi.MX.sym("positive_deviation" + name)
        self.negative_deviation = casadi.MX.sym("negative_deviation" + name)

    def set_objective(
        self,
    ) -> casadi.MX:
        return self.positive_deviation

    def set_constraints(
        self,
        packages: npt.NDArray,
    ) -> Tuple[npt.NDArray, casadi.MX, npt.NDArray]:
        lower_bound_list: List[float] = []
        upper_bound_list: List[float] = []
        constraint_list: List[casadi.MX] = []

        assert packages.size == self.allocations.numel()
        assert np.min(packages) > 0
        assert np.all(packages[:-1] <= packages[1:])

        # set deviation
        # Note: Deviation is already in positive-negative decomposition.
        lower_bound_list.append(0)
        constraint_list.append(
            (self.positive_deviation - self.negative_deviation)
            - (self.demand - casadi.dot(packages, self.allocations))
        )
        upper_bound_list.append(0)

        # set minimal_allocation
        lower_bound_list.append(0)
        constraint_list.append(
            casadi.dot(packages, self.allocations)
            - self.minimal_allocation_factor * self.demand
        )
        upper_bound_list.append(1e20)

        lower_bound = np.array(lower_bound_list)
        constraints = casadi.vertcat(*constraint_list)
        upper_bound = np.array(upper_bound_list)

        return lower_bound, constraints, upper_bound
