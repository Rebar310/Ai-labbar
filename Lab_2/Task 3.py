import numpy as np
import pickle 

##### Load the data here #####

data = "Task_3_data.pkl"
with open(data, "rb") as f:
    data = pickle.load(f)

##### Convert the data into numpy arrays here ########

data = np.array(data, dtype=float)

# sista kolumnen = y
X = data[:, :-1]
y = data[:, -1]

# antal features (kolumner i X)
num_features = X.shape[1]

# initiala vikter (måste matcha antal kolumner i X)
init_weights = np.zeros(num_features)

########## End of Data preperation ##############

learning_rate = 0.3
def cost_function(X, y, weights):
    predictions = X @ weights
    return np.mean((predictions - y) ** 2)


def cost_function_gd(X, y, weights):
    predictions = X @ weights
    return (2 / len(y)) * X.T @ (predictions - y)

# Adam
def Adam (X, y, init_weights, iterations):
    weights = init_weights.copy()
    # beta1, beta2 = 0.2, 0.2 (original values)
    beta1, beta2 = 0.9, 0.99 # from slides
    # epsilon = 1e-8 (original values)
    epsilon = 1e-2 # from slides
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    costs = []
    for t in range(1, iterations + 1):
        grad = cost_function_gd(X, y, weights)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        costs.append(cost_function(X, y, weights))
    return weights, costs

# RMSProp
def RMSProp(X, y, init_weights, iterations):
    weights = init_weights.copy()
    epsilon = 1e-8
    decay_rate = 0.1
    grad_accum = np.zeros_like(weights)
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        grad_accum = decay_rate * grad_accum + (1 - decay_rate) * grad ** 2
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon)
        weights -= adjusted_lr * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Momentum (GD with momentum)
def Momentum(X, y, init_weights, iterations):
    weights = init_weights.copy()
    v = np.zeros_like(weights)  # Velocity term
    momentum = 0.1
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        v = momentum * v - learning_rate * grad
        weights += v
        costs.append(cost_function(X, y, weights))
    return weights, costs

# AdaGrad
def AdaGrad(X, y, init_weights, iterations):
    weights = init_weights.copy()
    epsilon = 1e-8
    grad_accum = np.zeros_like(weights)
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        grad_accum += grad ** 2
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon)
        weights -= adjusted_lr * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Vanlig Gradient Descent
def GradientDescent(X, y, init_weights, iterations):
    weights = init_weights.copy()
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        weights -= learning_rate * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs




# ============== Test ========================

iterations = 20

optimizers = {
    "optimizer_1 (Adam)": Adam,
    "optimizer_2 (RMSProp)": RMSProp,
    "optimizer_3 (Momentum)": Momentum,
    "optimizer_4 (AdaGrad)": AdaGrad,
    "optimizer_5 (GD)": GradientDescent,
}

# Enkelt konvergenskriterium:
# "konvergerar" om förbättringen blir väldigt liten mot slutet
epsilon_conv = 0.001 #from slides

print("=== Resultat efter 20 iterationer ===")
for name, opt in optimizers.items():
    weights, costs = opt(X, y, init_weights, iterations)

    converged = False
    converged_at = None

    # leta efter första t där J_{t-1} - J_t <= epsilon
    for t in range(1, len(costs)):
        decrease = costs[t-1] - costs[t]
        if 0 <= decrease <= epsilon_conv:
            converged = True
            converged_at = t + 1  # iterationnummer (1-indexat)
            break

    print(f"{name}")
    print(f"  start cost: {costs[0]:.6f}")
    print(f"  end   cost: {costs[-1]:.6f}")
    if converged:
        print(f"  converged at iteration: {converged_at} (ε={epsilon_conv})")
    else:
        print(f"  not converged within {iterations} iterations (ε={epsilon_conv})")
    print()






