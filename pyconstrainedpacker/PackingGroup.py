from math import fabs
from typing import Dict, List, Tuple

import casadi
import numpy as np
import numpy.typing as npt


class PackingGroup:
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

        self.name: str = name
        self.demand: float = demand
        self.minimal_allocation_factor: float = minimal_allocation_factor
        self.number_of_package_sizes: int = number_of_package_sizes

        self.allocations: casadi.MX = casadi.MX.sym(
            "allocations_" + name, self.number_of_package_sizes
        )
        self.positive_deviation = casadi.MX.sym("positive_deviation" + name)
        self.negative_deviation = casadi.MX.sym("negative_deviation" + name)
        self.startindex: int
        self.endindex: int

        self.deviation_value: float = 1e20
        self.packed_numbers: List[float] = []

    def declare_variables_and_their_bounds(
        self, startindex: int
    ) -> Tuple[npt.NDArray, casadi.MX, npt.NDArray, List[bool], int]:
        self.startindex = startindex
        number_of_variables = 2 + self.number_of_package_sizes
        lower_bound_list: List[float] = (number_of_variables) * [0.0]
        upper_bound_list: List[float] = (number_of_variables) * [1e20]
        discrete = [False, False] + self.number_of_package_sizes * [True]
        variable_list: List[casadi.MX] = [
            self.negative_deviation,
            self.positive_deviation,
            self.allocations,
        ]

        lower_bound = np.array(lower_bound_list)
        variables = casadi.vertcat(*variable_list)
        upper_bound = np.array(upper_bound_list)

        self.endindex = startindex + number_of_variables
        return (lower_bound, variables, upper_bound, discrete, self.endindex)

    def set_objective(
        self,
    ) -> casadi.MX:
        c = 1
        alpha = 1e-5
        return c * self.negative_deviation + alpha * self.positive_deviation

    def set_constraints(
        self,
        package_sizes: npt.NDArray,
    ) -> Tuple[npt.NDArray, casadi.MX, npt.NDArray]:
        lower_bound_list: List[float] = []
        upper_bound_list: List[float] = []
        constraint_list: List[casadi.MX] = []

        assert package_sizes.size == self.allocations.numel()
        assert np.min(package_sizes) > 0
        assert np.all(package_sizes[:-1] <= package_sizes[1:])

        # set deviation
        # Note: Deviation is already in positive-negative decomposition.
        lower_bound_list.append(0)
        constraint_list.append(
            (self.positive_deviation - self.negative_deviation)
            - (casadi.dot(package_sizes, self.allocations) - self.demand)
        )
        upper_bound_list.append(0)

        # set minimal_allocation
        lower_bound_list.append(0)
        constraint_list.append(
            casadi.dot(package_sizes, self.allocations)
            - self.minimal_allocation_factor * self.demand
        )
        upper_bound_list.append(1e20)

        lower_bound = np.array(lower_bound_list)
        constraints = casadi.vertcat(*constraint_list)
        upper_bound = np.array(upper_bound_list)

        return lower_bound, constraints, upper_bound

    def save_variables_in_group(self, values: npt.NDArray) -> None:
        self.deviation_value = values[self.startindex] - values[self.startindex + 1]
        for i in range(self.startindex + 2, self.endindex):
            assert fabs((values[i] - int(round(values[i])))) < 1e-10
        self.packed_numbers = [
            int(value) for value in values[self.startindex + 2 : self.endindex]
        ]

    def print_packed_numbers(self, package_sizes: List[float]) -> None:
        results = self.create_results_dictionary(package_sizes)

        for key, value in results.items():
            print(f"{key}: {value}")
        # print(f"name: {self.name}")
        # print("Packed units by size:")
        # for i in range(len(package_sizes)):
        #     print(f"{package_sizes[i]}: {self.packed_numbers[i]}")
        # print(f"Deviation: {self.deviation_value}")

    def create_results_dictionary(
        self, package_sizes: List[float]
    ) -> Dict[str | float, str | float | int]:
        results_json: Dict[str | float, str | float | int] = dict()
        results_json["name"] = self.name
        for i in range(len(package_sizes)):
            results_json[package_sizes[i]] = self.packed_numbers[i]
        results_json["deviation"] = self.deviation_value
        return results_json
