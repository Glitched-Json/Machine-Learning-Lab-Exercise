from random import random
from sklearn.cluster import MiniBatchKMeans
import cv2 as cv

from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

descs = []
for img in train_X:
    sift = cv.SIFT_create()
    keyPoints, desc = sift.detectAndCompute(img, None)
    descs.append(desc)


kmeans = MiniBatchKmeans(n_clusters=100, batch_size=train_X.size * 4, verbose=1).fit(descs)

# kmeans finds word pool

# extract class in kmeans of sift desc of images
# use as words in Bag Of Words, #similar to ex2

index = random.randrange(0, len(train_X))
print(train_X[index])
