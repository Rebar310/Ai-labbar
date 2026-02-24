# This imports tools/ libaries
import matplotlib.pyplot as plt  # for plotting and visualization
import numpy as np  # used for numerical operations
import pandas as pd  # panda is for table data
from sklearn.ensemble import RandomForestClassifier #Used for RandomForest
from sklearn.svm import SVC  # support vector classifier

# 1. load the data (dataframe) from file.. with panda----------------
df = pd.read_pickle('Lab1_Task4_data.pkl') # load dataset into panda dataframe
X = df[['Tissue Texture Score', 'Tissue Density Score']].values #convert to numpy array
y = df['Diagnosis'] #select the target column (class labels)

# 3. Fitting a model to the training data ----------------
svm_clf = SVC(kernel='poly', degree=3 , C =1 , coef0=1) # create support vector classifier
svm_clf.fit(X, y) #train the classifier to fit input data

# --------------------------- code I added
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X, y)
# ------------------------------

# create grid
x_texture_min, x_texture_max = int(min(df['Tissue Texture Score'])), int(max(df['Tissue Texture Score'])) #find min and max value of texture feature
x_density_min, x_density_max = int(min(df['Tissue Density Score'])), int(max(df['Tissue Density Score'])) #find max and min value of Density

# 4. Transforming the test data to find the decision boundary
xx_texture, xx_density = np.meshgrid(np.linspace(x_texture_min, x_texture_max, num=100), np.linspace(x_density_min, x_density_max, num=100))
xx_decision_boundary = np.stack((xx_texture.flatten(), xx_density.flatten()), axis=1) # convert to list of coordinates (x,y)
Z_SVM = svm_clf.predict(xx_decision_boundary) #predict class of the coordinate
Z_SVM = Z_SVM.reshape(xx_texture.shape) # reshape prediction so it fits the grid shape
Z_RF  = rf_clf.predict(xx_decision_boundary).reshape(xx_texture.shape)  # I added

# 5. Visualizing the decision boundary -------------------
# plt.figure() # old, single plot, used before
fig, axes = plt.subplots(2, 1, figsize=(7, 10))  # two plots ( I added)

axes[0].contourf(xx_texture, xx_density, Z_SVM, alpha=0.4)
axes[1].contourf(xx_texture, xx_density, Z_RF, alpha=0.4)

# 2. Visualize the data ----------------------------------

# <<<<<<<<<<<< Everything for the SVM plot >>>>>>>>>>>>>>>>>>>>>> changed plt. to axes[1]
axes[0].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')
axes[0].set_title("SVM") # using set_title instead of title because of subplots
axes[0].set_xlabel("Tissue Texture Score")
axes[0].set_ylabel("Tissue Density Score")

# <<<<<<<<<<< Random Forest Plot >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
axes[1].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Tissue Texture Score")
axes[1].set_ylabel("Tissue Density Score")

plt.tight_layout()
plt.show()
