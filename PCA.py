#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#reading data
d = pd.read_csv('PCA_practice_dataset.csv')
data = d.to_numpy()

print(data.shape)

#creating PCA object
pca = PCA()
pca.fit(data)
data_pca = pca.transform(data)

#calculating cumulative variance
cu_var = np.cumsum(pca.explained_variance_ratio_)

#initializing the thresholds
th = [i for i in np.arange(0.90,0.98,0.01)]
thresholds = [round(i,2) for i in th]

num_comp = [np.argmax(cu_var>t) for t in thresholds]

#printing out the thresholds and number of components
for comp, t in zip(num_comp, thresholds):
    print("Number of Components required for Threshold value of ",t," are: ",comp)

#plotting the data
plt.plot(num_comp, thresholds, linewidth = 3)
plt.title("Scree plot")
plt.xlabel("Principal Components")
plt.ylabel("Threshold value")
plt.show()

#output!!!!!!!!!!!!!