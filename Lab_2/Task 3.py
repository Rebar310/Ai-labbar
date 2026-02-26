import numpy as np # Importerar NumPy för numeriska beräkningar
import pickle # Importerar pickle för att kunna läsa in sparade Python-objekt
import matplotlib.pyplot as plt

##### Load the data here #####

data = "Task_3_data.pkl" # Filnamnet (sträng) till datasetet som ska laddas
with open(data, "rb") as f: # Öppnar filen i binärt läsläge ("rb" = read binary)
    data = pickle.load(f) # Läser in (deserialiserar) objektet från filen och sparar i variabeln data

##### Convert the data into numpy arrays here ########

data = np.array(data, dtype=float) # Konverterar datan till en NumPy-array och tvingar datatyp till float

# sista kolumnen = y
X = data[:, :-1] # Tar alla rader och alla kolumner utom sista => features (in-data)
y = data[:, -1] # Tar alla rader men bara sista kolumnen => target (y)

# antal features (kolumner i X)
num_features = X.shape[1] # Antal kolumner i X = antal features

# initiala vikter (måste matcha antal kolumner i X)
init_weights = np.zeros(num_features) # Skapar en viktvektor med num_features st nollor

########## End of Data preperation ##############

learning_rate = 0.3 # Steglängd (hur stora uppdateringar i vikterna varje iteration)

def cost_function(X, y, weights):
    # Beräknar modellens prediktioner med linjär modell: y_hat = X @ weights
    predictions = X @ weights
    # Returnerar Mean Squared Error (MSE): medelvärdet av (pred - y)^2
    return np.mean((predictions - y) ** 2)


def cost_function_gd(X, y, weights):
    # Beräknar prediktionerna
    predictions = X @ weights
    # Returnerar gradienten till MSE med avseende på weights:
    # (2/n) * X^T * (predictions - y)
    return (2 / len(y)) * X.T @ (predictions - y)

# Adam
def Adam (X, y, init_weights, iterations):
    weights = init_weights.copy() # Kopierar initialvikter så att originalet inte ändras

    #beta1, beta2 = 0.2, 0.2 #(original values)
    beta1, beta2 = 0.09, 0.09 # changed values
    #beta1, beta2 = 0.9, 0.99 # changed values

    epsilon = 1e-8 # Litet tal för att undvika division med 0 (och stabilisera)
    m = np.zeros_like(weights) # Första momentet (running average av gradienten)
    v = np.zeros_like(weights) # Andra momentet (running average av gradient^2)
    costs = []   # Lista för att spara kostnad efter varje iteration

    for t in range(1, iterations + 1):
        grad = cost_function_gd(X, y, weights) # Beräknar gradienten för nuvarande weights
        m = beta1 * m + (1 - beta1) * grad # Uppdaterar första momentet m
        v = beta2 * v + (1 - beta2) * (grad ** 2) # Uppdaterar andra momentet v
        m_hat = m / (1 - beta1) # Bias-korrigering av m
        v_hat = v / (1 - beta2) # Bias-korrigering av v
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) # Adam-uppdatering av vikter
        costs.append(cost_function(X, y, weights)) # Sparar kostnaden efter uppdateringen
    return weights, costs

# RMSProp
def RMSProp(X, y, init_weights, iterations):
    weights = init_weights.copy() # Kopierar initialvikter
    epsilon = 1e-8 # Litet tal för numerisk stabilitet
    decay_rate = 0.1 # Hur snabbt historiken "glöms" (running average)
    grad_accum = np.zeros_like(weights) # Ackumulator för running average av gradient^2
    costs = [] # Lista för kostnader

    for _ in range(iterations): # Kör iterations antal varv
        grad = cost_function_gd(X, y, weights) # Gradient
        grad_accum = decay_rate * grad_accum + (1 - decay_rate) * grad ** 2 # Uppdaterar ackumulerad grad^2
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon) # Anpassad learning rate per parameter
        weights -= adjusted_lr * grad # Uppdaterar vikter med RMSProp
        costs.append(cost_function(X, y, weights)) # Sparar kostnad
    return weights, costs

# Momentum (GD with momentum)
def Momentum(X, y, init_weights, iterations):
    weights = init_weights.copy() # Kopierar initialvikter
    v = np.zeros_like(weights)  # Velocity term (ackumulerar riktning/hastighet)
    momentum = 0.1 # Momentumfaktor (hur mycket av tidigare v som behålls)
    costs = [] # Lista för kostnader

    for _ in range(iterations): # Loopar antal iterationer
        grad = cost_function_gd(X, y, weights) # Beräknar gradient
        v = momentum * v - learning_rate * grad # Uppdaterar velocity (rör sig i riktning mot minskande kostnad)
        weights += v # Uppdaterar vikter med velocity
        costs.append(cost_function(X, y, weights)) # Sparar kostnaden
    return weights, costs

# AdaGrad
def AdaGrad(X, y, init_weights, iterations):
    weights = init_weights.copy() # Kopierar initialvikter
    epsilon = 1e-8 # Stabilitetsterm för division
    grad_accum = np.zeros_like(weights) # Ackumulator för summan av grad^2 över tid
    costs = [] # Lista för kostnader

    for _ in range(iterations): # Loopar antal iterationer
        grad = cost_function_gd(X, y, weights) # Gradient
        grad_accum += grad ** 2 # Lägger till grad^2 (växer över tid)
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon) # Anpassar learning rate (minskar med tiden)
        weights -= adjusted_lr * grad # Uppdaterar vikter
        costs.append(cost_function(X, y, weights)) # Sparar kostnad
    return weights, costs

# Vanlig Gradient Descent
def GradientDescent(X, y, init_weights, iterations):
    weights = init_weights.copy() # Kopierar initialvikter
    costs = [] # Lista för kostnader

    for _ in range(iterations): # Loopar antal iterationer
        grad = cost_function_gd(X, y, weights) # Gradient
        weights -= learning_rate * grad # Standard GD-uppdatering
        costs.append(cost_function(X, y, weights)) # Sparar kostnad
    return weights, costs


# ================ Test ( Technique 1) =====================
iterations = 20 # Antal iterationer att köra varje optimerare

optimizers = { # Dictionary som mappar namn -> funktionsreferens
   "optimizer_1 (Adam)": Adam,
   "optimizer_2 (RMSProp)": RMSProp,
    "optimizer_3 (Momentum)": Momentum,
    "optimizer_4 (AdaGrad)": AdaGrad,
    "optimizer_5 (GD)": GradientDescent,
 }

plt.figure(figsize=(8,6))

for name, opt in optimizers.items():
    weights, costs = opt(X, y, init_weights, iterations)
    plt.plot(costs, label=name)

plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Learning Curves for All Optimizers")
plt.legend()
plt.grid(True)
plt.show()

# ============== Test (Technique 2) ========================

# iterations = 100 # Antal iterationer att köra varje optimerare

# optimizers = { # Dictionary som mappar namn -> funktionsreferens
#     "optimizer_1 (Adam)": Adam,
#     "optimizer_2 (RMSProp)": RMSProp,
#     "optimizer_3 (Momentum)": Momentum,
#     "optimizer_4 (AdaGrad)": AdaGrad,
#     "optimizer_5 (GD)": GradientDescent,
# }

# # Enkelt konvergenskriterium:
# # "konvergerar" om förbättringen blir väldigt liten mot slutet
# epsilon_conv = 0.001 #from slides

# print("=== Resultat efter ",iterations," iterationer ===")
# for name, opt in optimizers.items():  # Loopar över varje optimerare (namn och funktion)
#     weights, costs = opt(X, y, init_weights, iterations) # Kör optimeraren och får tillbaka slutvikter + kostnadshistorik

#     converged = False # Flagga för om den konvergerade
#     converged_at = None # Vid vilken iteration den konvergerade (om den gjorde det)

#     # leta efter första t där J_{t-1} - J_t <= epsilon
#     for t in range(1, len(costs)):  # Startar vid 1 eftersom vi jämför costs[t-1] med costs[t]
#         decrease = costs[t-1] - costs[t] # Hur mycket kostnaden minskade mellan två iterationer
#         if 0 <= decrease <= epsilon_conv: # Om den minskar men bara pyttelite => anses konvergerat
#             converged = True # Sätt flaggan
#             converged_at = t + 1  # iterationnummer (1-indexat)
#             break

#     print(f"{name}")
#     print(f"  start cost: {costs[0]:.6f}")
#     print(f"  end   cost: {costs[-1]:.6f}")
#     if converged:
#         print(f"  converged at iteration: {converged_at} (ε={epsilon_conv})")
#     else:
#         print(f"  not converged within {iterations} iterations (ε={epsilon_conv})")
#     print()

# =======================================================




