from typing import List, Tuple

import casadi
import numpy as np
import numpy.typing as npt

from .Group import Group


class Shipment:
    def __init__(
        self, package_sizes: List[float], number_of_packages_per_size: List[int]
    ) -> None:
        assert len(package_sizes) == len(number_of_packages_per_size)
        for size in package_sizes:
            assert size > 0
        for number in number_of_packages_per_size:
            assert number > 0
        self.package_sizes = np.array(package_sizes)
        self.package_numbers = np.array(number_of_packages_per_size)

    def set_constraints(
        self, groups: List[Group]
    ) -> Tuple[npt.NDArray, casadi.MX, npt.NDArray]:
        """Constrain the allocations to the number of packages available."""

        constraints = sum([group.allocations for group in groups])

        return (
            np.array(len(self.package_sizes) * [-1e20]),
            constraints,
            self.package_numbers,
        )
