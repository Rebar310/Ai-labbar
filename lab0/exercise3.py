import numpy as np

def estimate_pi( N ):

    # Sample of N points which all lie in a square
    points = np.random.random((N,2))
    x = points[:,0]
    y = points[:,1]

    # Count points inside circle
    inside_circle = (x - 0.5)**2 + (y-0.5)**2 < 0.5**2
    M = inside_circle.sum()

    # pi = 4 * lamda( M / N )
    return 4.0 * M / N

# Bonus exercise -------------------------------------
N_values = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]

pi_values = []
for N in N_values:
    pi_values.append(estimate_pi(N))

errors = []
for i in range(len(pi_values) - 1):
    error = abs(pi_values[i + 1] - pi_values[i])
    errors.append(error)

print("N\t\tpi(N)\t\tError")
for i in range(len(errors)):
    print(f"{N_values[i]:<8}\t{pi_values[i]:.6f}\t{errors[i]:.6f}")

# ---------------------
print("-----------------")
print(estimate_pi(30000))
# N needs to have at least have N value that is around 30 000 to have the .14 in the decimals
