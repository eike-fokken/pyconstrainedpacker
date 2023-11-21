# d = dict()

# d[5.3] = 9
# d[2.3] = 4

# print("presort:")
# for key, value in d.items():
#     print(f"{key}: {value}")

# print("postsort:")
# for key, value in sorted(d.items()):
#     print(f"{key}: {value}")

# print(d)
import casadi

# import numpy as np

x = casadi.MX.sym("x")

y = casadi.MX.sym("y", 2)

mylist = [x, y]


final_casadi_variable = casadi.vertcat(*mylist)

print(final_casadi_variable.size())
