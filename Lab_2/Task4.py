import numpy as np
import pickle
import matplotlib.pyplot as plt


# ---------- 1) Load data ----------
with open("Task_4_data.pkl", "rb") as f:
    df = pickle.load(f)  # brukar vara en pandas DataFrame

data = df.to_numpy(dtype=float)

# Last column = y
X_raw = data[:, :-1]
y = data[:, -1]

feature_names = list(df.columns[:-1])
target_name = df.columns[-1]

n_samples, n_features = X_raw.shape
print(f"Samples: {n_samples}, Features: {n_features}")
print("Features:", feature_names)
print("Target:", target_name)

# ---------- (Rekommenderat) Standardisera features för stabilare GD ----------
mu = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0) + 1e-12
X_std = (X_raw - mu) / sigma

# Lägg till bias/intercept (kolumn med 1:or)
X = np.c_[np.ones(n_samples), X_std]

# Initiala vikter (måste matcha antal kolumner i X)
init_weights = np.zeros(X.shape[1])

# ---------- Hyperparametrar ----------
learning_rate = 0.1
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

# ---------- 3) Träna + hitta konvergens-iteration ----------
weights, costs = GradientDescent(X, y, init_weights, max_iter)

converged_at = None
for t in range(1, len(costs)):
    decrease = costs[t-1] - costs[t]   # J_{t-1} - J_t
    if 0 <= decrease <= epsilon_conv:
        converged_at = t + 1  # 1-indexad iteration
        break

print("\n=== RESULTS ===")
print(f"Final MSE: {costs[-1]:.6f}")
if converged_at is None:
    print(f"Convergence: NOT reached within {max_iter} iterations (ε={epsilon_conv})")
else:
    print(f"Convergence: reached at iteration {converged_at} (ε={epsilon_conv})")

# ---------- 4) Presentera slutlig linjär modell (i ORIGINAL SKALA) ----------
# Modellen du tränade är på standardiserad form:
# y = w0 + sum_i w_i * (x_i - mu_i)/sigma_i
# Konvertera till rå (original) skala:
w0_std = weights[0]
w_std = weights[1:]

coef_raw = w_std / sigma
intercept_raw = w0_std - np.sum((w_std * mu) / sigma)

print("\nFinal linear regression model (original scale):")
print(f"{target_name} = {intercept_raw:.6f}", end="")
for fname, c in zip(feature_names, coef_raw):
    print(f" + ({c:.6f})*{fname}", end="")
print("\n")

# ---------- 5) Plot learning curve + markera konvergens ----------

iters = np.arange(1, len(costs) + 1)  # 1-indexade iterationer

max_show = 50  # antal iterationer du vill visa

plt.figure(figsize=(9,5))
plt.plot(iters[:max_show], costs[:max_show], label="MSE (first 50)")

if converged_at is not None and converged_at <= max_show:
    plt.axvline(converged_at, linestyle="--", label=f"Convergence @ {converged_at}")
    plt.scatter([converged_at], [costs[converged_at - 1]])

plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Learning curve (first 50 iterations)")
plt.grid(True)
plt.legend()
plt.show()

