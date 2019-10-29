import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing
import random

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'

    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)

    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images
    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    train_categories = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    train_categories.fit(train_features, train_labels)
    predicted_categories = train_categories.predict(test_features)

    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicte d category for each test feature.
    predicted_categories = []
    if is_linear:
        train = svm.LinearSVC(C=svm_lambda).fit(train_features, train_labels)
        predicted_categories = train.predict(test_features)
    else:
        train = svm.SVC(kernel='rbf',C=svm_lambda,gamma='scale').fit(train_features, train_labels)
        predicted_categories = train.predict(test_features)
    
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size].
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    dim = (target_size, target_size)
    resized = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    output_image = cv2.normalize(resized, None, alpha=1, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an
    # integer, and N is the size of the testing set.

    correct_pred = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct_pred += 1
    accuracy = correct_pred / len(predicted_labels)
    return accuracy * 100


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the
    # number of descriptors you store per image.
    # SIFT
    descriptors = []
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=64)
    surf = cv2.xfeatures2d.SURF_create(extended=False)
    orb = cv2.ORB_create(nfeatures=64)
    for images in train_images:
        if feature_type == 'sift':
            kp, des = sift.detectAndCompute(images, None)
            if np.shape(des)[0] < 30: 
                for x in range(np.shape(des)[0]):
                    descriptors.append(des[x])
            else:
                for _ in range(30):
                    descriptors.append(des[random.randrange(np.shape(des)[0])])
        elif feature_type == 'surf':
            kp, des = surf.detectAndCompute(images, None)
            if np.shape(des)[0] < 30: 
                for x in range(np.shape(des)[0]):
                    descriptors.append(des[x])
            else:
                for _ in range(30):
                    descriptors.append(des[random.randrange(np.shape(des)[0])])
        elif feature_type == 'orb':
            kp, des = orb.detectAndCompute(images, None)
            if des is None:
                continue
            if np.shape(des)[0] < 30: 
                for x in range(np.shape(des)[0]):
                    descriptors.append(des[x])
            else:
                for _ in range(30):
                    descriptors.append(des[random.randrange(np.shape(des)[0])])

    if clustering_type == 'kmeans':
        clusters = cluster.KMeans(n_clusters=dict_size).fit(descriptors)
        vocabulary = clusters.cluster_centers_
    elif clustering_type == 'hierarchical':
        vocabulary = []
        clusters = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)
        label_groups = []
        for x in range(dict_size):
            label_groups.append([])
        for x in range(len(descriptors)):
            label_groups[clusters.labels_[x]].append(descriptors[x])
        for x in range(dict_size):
            total = np.array([0] * len(descriptors[0]), dtype=np.float64)
            for y in range(len(label_groups[x])):
                total += label_groups[x][y]
            total /= len(label_groups[x])
            vocabulary.append(total)
    
    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    Bow = []
    if feature_type == 'sift':
      	sift = cv2.xfeatures2d.SIFT_create()
      	kp, des = sift.detectAndCompute(image, None)
        labels = [0] * len(vocabulary)
        for d in des:
          	e_min = 10000
          	min_index = 0
          	for v in range(len(vocabulary)):
            	e_dist = np.linalg.norm(d - vocabulary[v])
            	if e_min > e_dist:
            		min_index = v
                	e_min = e_dist
          	labels[min_index] += 1
        Bow = labels
    elif feature_type == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(image, None)
        labels = [0] * len(vocabulary)
        for d in des:
          	e_min = 10000
          	min_index = 0
          	for v in range(len(vocabulary)):
            	e_dist = np.linalg.norm(d - vocabulary[v])
            	if e_min > e_dist:
            		min_index = v
                	e_min = e_dist
          	labels[min_index] += 1
        Bow = labels
	elif feature_type == 'orb':
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(image, None)
        if des is None:
        	return [0] * len(vocabulary)
        labels = [0] * len(vocabulary)
        for d in des:
          	e_min = 10000
          	min_index = 0
          	for v in range(len(vocabulary)):
            	e_dist = np.linalg.norm(d - vocabulary[v])
            	if e_min > e_dist:
            		min_index = v
                	e_min = e_dist
          	labels[min_index] += 1
        Bow = labels
    sums = 0
    for x in range(len(Bow)):
        sums += Bow[x]
    for x in range(len(Bow)):
        Bow[x] = Bow[x]/sums
    return Bow


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors
    train8 = []
    train16 = []
    train32 = []
    test8 = []
    test16 = []
    test32 = []
    for images in train_features:
        train8.append((imresize(images, 8)).flatten())
        train16.append((imresize(images, 16)).flatten())
        train32.append((imresize(images, 32)).flatten())

    for imgTest in test_features:
        test8.append((imresize(imgTest, 8)).flatten())
        test16.append((imresize(imgTest, 16)).flatten())
        test32.append((imresize(imgTest, 32)).flatten())

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy,
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    classResult = []
    k8 = []
    # for 8x8
    for k in [1, 3, 6]:
        t1 = time.time()
        k8 = KNN_classifier(train8, train_labels, test8, k)
        classResult.append(reportAccuracy(test_labels, k8))
        classResult.append(time.time() - t1)

    # for 16x16
    for k in [1, 3, 6]:
        t1 = time.time()
        k16 = KNN_classifier(train16, train_labels, test16, k)
        classResult.append(reportAccuracy(test_labels, k16))
        classResult.append(time.time() - t1)

    # for 32x32
    for k in [1, 3, 6]:
        t1 = time.time()
        k32 = KNN_classifier(train32, train_labels, test32, k)
        classResult.append(reportAccuracy(test_labels, k32))
        classResult.append(time.time() - t1)

    return classResult
