import numpy as np
import pickle
import matplotlib.pyplot as plt
#from sklearn.datasets import make_regression    #för regression dataset


# ---------- 1) Load data ----------
data_path="Task_4_data.pkl"
with open(data_path, "rb") as file:
    df = pickle.load(file)  # brukar vara en pandas DataFrame

data = np.array(df,dtype=float)

# Last column = y
X= data[:, :-1]
y = data[:, -1]

m, n = X.shape
print(f"Dataset loaded: {m} samples, {n} features")
y=y.reshape(-1,1)  # gör y till en kolumnvektor


# Lägg till bias/intercept (kolumn med 1:or)
X_b= np.c_[np.ones((m,1)), X]  # storlek (m,n+1)


# Initiala vikter (måste matcha antal kolumner i X)
init_weights = np.zeros((n+1,1))  # n+1 eftersom vi har en bias-kolumn också
# ---------- Hyperparametrar ----------
learning_rate = 0.1
epsilon_conv = 0.001   # "Automatic convergence test" från sliden
max_iter = 100
prev_mse = float('inf')  # För att spåra MSE-förändring
costs=[]  # För att spara kostnader för plotten
konvergens_iter = None  # För att spara när konvergens inträffar
#GradientDescent-funktion och kostnadsfunktioner återanvänds från Task 3, så de är kommenterade här.
for i in range(max_iter):
    # Beräkna kostnadsgradient,
    y_pre = np.dot(X_b, init_weights)  # prediktioner

    #beräkna fel och mse
    error = y_pre - y
    mse = np.mean(error ** 2)

    #kovergens test
    if abs(prev_mse - mse) <= epsilon_conv:
        print(f"konvergens efter {i+1} iterationer, MSE: {mse:.6f}")
        konvergens_iter = i + 1
        break
    prev_mse = mse
    #beräkna gradient
    grad = (2 / m) * np.dot(X_b.T, error)  
    
    # Uppdatera vikter
    init_weights -= learning_rate * grad

    costs.append(mse)  # Spara kostnaden för varje iteration
#plotta kostnadsfunktion
plt.figure(figsize=(8, 5))
plt.plot(costs, marker='o')
plt.title('MSE över iterationer')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.grid()
if konvergens_iter is not None:
    plt.axvline(x=konvergens_iter-1, color='r', linestyle='--', label='Konvergens')
    plt.legend()

plt.show()

#presentation av resultat, slutgiltiga model (y= w0 + w1*x1 + w2*x2 + ... + wn*xn) och slutlig MSE
print("\n" + "="*30)
print("FINAL LINEAR REGRESSION MODEL")
print("="*30)
w0= init_weights[0,0]  # bias
weights= init_weights[1:].flatten()  # övriga vikter
y_eq = f"y = {w0:.4f}" + "".join([f" + ({weights[i]:.4f} * x{i+1})" for i in range(len(weights))])

print(f"Y-ekvation: {y_eq}")

print("Finala vikter (inklusive bias):")
print(init_weights.flatten())
print(f"Slutlig MSE: {mse:.6f}")


