from mlxtend.data import loadlocal_mnist
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, precision_score, recall_score
print ("        *** Loading Data..... *** ")
################### Loading Data #################
(train_x, train_y) = loadlocal_mnist(images_path='/content/drive/MyDrive/train-images.idx3-ubyte', 
                                     labels_path='/content/drive/MyDrive/train-labels.idx1-ubyte')

(test_x, test_y) = loadlocal_mnist(images_path='/content/drive/MyDrive/t10k-images.idx3-ubyte', 
                                   labels_path='/content/drive/MyDrive/t10k-labels.idx1-ubyte')

print ("        *** Applaying HOG *** ")
#####################  HOG  ####################
list_hog_fd = []
list_hog_image = []
for feature in train_x:
  fd, hog_image = hog(feature.reshape(28,28), orientations=8, pixels_per_cell=(4,4),
                      cells_per_block=(1, 1), visualize=True, transform_sqrt=True)
  list_hog_fd.append(fd)
  list_hog_image.append(hog_image)
hog_features = np.array(list_hog_fd)


t_list_hog_fd = []
t_list_hog_image = []
for test_feature in test_x:
  t_fd, t_hog_image = hog(test_feature.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualize=True,transform_sqrt=True)
  t_list_hog_fd.append(t_fd)
  t_list_hog_image.append(t_hog_image)
t_hog_features = np.array(t_list_hog_fd)
####### plot sample of hogged images #######
plt.axis("off")
plt.imshow(list_hog_image[2], cmap="gray")
plt.show()
print ("        *** Fit & Test KNN *** ")
##################### KNN ##################
knn_model = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
knn_model.fit(hog_features, train_y)
knn_prediction = knn_model.predict(t_hog_features)

print ("        *** Fit & Test SVM *** ")
##################### SVM ##################  
svm_model = svm.SVC()
svm_model.fit(hog_features, train_y)
svm_prediction = svm_model.predict(t_hog_features)

print ("        *** Fit & Test ANN *** ")
##################### ANN ################## 
ann_model = Sequential()    
ann_model.add(Dense(392, input_dim=392, activation='relu'))
ann_model.add(Dense(56,activation='relu'))
ann_model.add(Dense(28,activation='relu'))
ann_model.add(Dense(14,activation='relu'))
ann_model.add(Dense(10,activation='softmax'))

ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ann_model.fit(hog_features, train_y, epochs=30)
ann_prediction = ann_model.predict(t_hog_features)
ann_prediction = np.argmax(ann_prediction, axis=1)

print ("        *** Evaluation & Comparing models *** ")
############## Evaluation & Comparisons #####################
def evaluation_accuracies(labels_list, prediction_list):
  print("Accuracy_Score = ", accuracy_score(labels_list, prediction_list) * 100, "%")
  print("R2_score = ", r2_score(labels_list, prediction_list) * 100, "%")
  print("Confusion matrix = \n", confusion_matrix(labels_list, prediction_list))
  print("precision = ", precision_score(labels_list, prediction_list, average=None) * 100)
  print("Recall = ", recall_score(labels_list, prediction_list, average=None) * 100)
  print(classification_report(labels_list, prediction_list))


def comparisons(labels_list, knn_prediction, svm_prediction, ann_prediction ):
  print("--------------------------------------------")
  print("KNN accuracies: \n")
  evaluation_accuracies(labels_list, knn_prediction)
  print("--------------------------------------------")
  print("SVM accuracies: \n")
  evaluation_accuracies(labels_list, svm_prediction)
  print("--------------------------------------------")
  print("ANN accuracies: \n")
  evaluation_accuracies(labels_list, ann_prediction)

comparisons(test_y, knn_prediction, svm_prediction, ann_prediction)