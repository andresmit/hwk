import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

#data
digits = datasets.load_digits()

#print(digits.data)
#print(digits.target)
#print(digits.images[0])
clf = svm.SVC(gamma = 0.0001, C=100)

print(len(digits.data))

#trenn
x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

#ennustus
print("prediction:", clf.predict(digits.data[-50]))
plt.imshow(digits.images[-50], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
