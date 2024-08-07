from typing import Dict, List, Tuple

import casadi
import numpy as np
import numpy.typing as npt

from .PackingGroup import PackingGroup


class ArticleShipment:
    def __init__(self, shipped_packages_by_size: Dict[float, int]) -> None:
        for size, number in shipped_packages_by_size.items():
            assert size > 0
            assert number >= 0
        self.package_sizes = np.array(list(shipped_packages_by_size.keys()))
        self.package_count = np.array(list(shipped_packages_by_size.values()))

    def set_constraints(
        self, groups: List[PackingGroup]
    ) -> Tuple[npt.NDArray, casadi.MX, npt.NDArray]:
        """Constrain the allocations to the number of packages available."""

        constraints = sum([group.allocations for group in groups])

        return (
            np.array(self.package_sizes.size * [0]),
            constraints,
            self.package_count,
        )
