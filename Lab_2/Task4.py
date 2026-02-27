import numpy as np
import pickle
import matplotlib.pyplot as plt


# ---------- 1) Load data ----------
data = "Task_3_data.pkl" # Filnamnet (sträng) till datasetet som ska laddas
with open(data, "rb") as f: # Öppnar filen i binärt läsläge ("rb" = read binary)
    data = pickle.load(f) # Läser in (deserialiserar) objektet från filen och sparar i variabeln data

##### Convert the data into numpy arrays here ########

data = np.array(data, dtype=float) # Konverterar datan till en NumPy-array och tvingar datatyp till float

# Last column = y
X_raw = data[:, :-1]
y = data[:, -1]

n_samples = len(y)

# Lägg till bias/intercept
X = np.c_[np.ones(n_samples), X_raw]

# Initiala vikter (måste matcha antal kolumner i X)
init_weights = np.zeros(X.shape[1])

# ---------- Hyperparametrar ----------
learning_rate = 0.6
epsilon_conv = 0.001   # "Automatic convergence test" från sliden
max_iter = 100000

# ---------- 2) Återanvända Task 3-funktioner ----------
def cost_function(X, y, weights):
    predictions = X @ weights
    return np.mean((predictions - y) ** 2)

def cost_function_gd(X, y, weights):
    predictions = X @ weights
    return (2 / len(y)) * X.T @ (predictions - y)

# Vanlig Gradient Descent (återanvänd från Task 3)
def GradientDescent(X, y, init_weights, iterations):
    weights = init_weights.copy()
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        weights -= learning_rate * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs

# ---------- 3) Träna + 4) Konvergens ----------
weights, costs = GradientDescent(X, y, init_weights, max_iter)

converged_at = None
for t in range(1, len(costs)):
    decrease = costs[t - 1] - costs[t]
    if 0 <= decrease <= epsilon_conv:
        converged_at = t + 1
        break

print("\nFinal MSE:", costs[-1])
print("Converged at iteration:", converged_at)

# ---------- 4) Slutlig modell ----------
print("\nFinal linear regression model:")
print("y =", weights[0], end="")
for i, coef in enumerate(weights[1:], start=1):
    print(f" + ({coef})*x{i}", end="")
print()

# _____________ Plot Cost _________________

iterations = 25 # Antal iterationer att köra varje optimerare

plt.figure(figsize=(8,6))

weights, costs = GradientDescent(X, y, init_weights, iterations)
plt.plot(costs)

plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Learning Curves for All Optimizers")
plt.legend()
plt.grid(True)
plt.show()