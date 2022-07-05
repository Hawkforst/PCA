import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from LDA import LDA

data = datasets.load_iris()
X = data.data
y = data.target

lda = LDA(2)

lda.fit(X,y)

X_projected = lda.transform(X)

print(f'Shape of X: {X.shape}')
print(f'Shape of transformed X: {X_projected.shape}')

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.colorbar()
plt.show()

data = datasets.load_breast_cancer()
lda_brst = LDA(3)
lda_brst.fit(data['data'], data['target'])
reduced_data = lda_brst.transform(data['data'])
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(projection='3d')

ax.scatter3D(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2],c=data['target'])
plt.show()