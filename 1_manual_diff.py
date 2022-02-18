def derivative_of_objective(x):
    return 2*x # derivative of x^2


curr = 327.52
for _ in range(1000):
    curr = curr - 0.01*derivative_of_objective(curr)

print(curr)
