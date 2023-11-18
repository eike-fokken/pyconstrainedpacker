from typing import Dict, List

import numpy as np


class Supply:
    def __init__(self, shipment: Dict[float, int]) -> None:
        package_size_list: List[float] = []
        package_number_list: List[int] = []
        for key, value in sorted(shipment.items()):
            package_size_list.append(key)
            package_number_list.append(value)

        self.package_sizes = np.array(package_size_list)
        self.package_numbers = np.array(package_number_list)
