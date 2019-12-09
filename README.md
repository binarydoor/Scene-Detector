# Scene Detector

![Catalogue Samples](https://github.com/binarydoor/Scene-Detector/raw/master/calatolgue_samples.png)


utils.py

	load_data - This function loads the test images and train images

	KNN_classifier - This takes the train images, train labels, test images,
					 and number of neighbors to perform KNN classifier using
					 sklearn.neighbors.KNeighborsClassifier[1] and return how
					 the test images correspond to the classifier.
	
	SVM_classifier - This takes train images, train labels, test images, boolean,
					 and svm lambda to perform SVM classifier using sklearn.svm.SVC[2]
					 to make the classifier with the train images and train labels
					 and return how the test images correspond to the classifier. 
	
	imresize - This resizes the input image to be target_size x target_size and 
			   return the normalized[3] image.

	reportAccuracy - This returns how the predicted labels are performing with
					 return the normalized[3] image

	imresize - This resizes the input image to be target_size x target_size and 
			   return the normalized[3] image

	reportAccuracy - This returns how the predicted labels are performing with
					 respect to the true label

	buildDict - This creates descriptors corresponding to the feature type and put
				the descriptors of all images into a clustering type (Kmeans[4],
				Agglomerative[5]) and return the vocabulary with specified size 
				(dict_size). Since the sklearn.cluster.AgglomerativeClustering does
				not have cluster_centers_, we needed to manually set the centroid 
				for this clustering type

	computeBow - This returns the computed BOW (Bag of Words) for a given image. We
				 first used feature to make descriptor of the given image and use 
				 that to create histogram and return the normalized histogram

	tinyImages - This resizes the train images to 8x8 16x16 and 32x32 and return the 
				 accuracy and runtime when the train images are fed to the KNN 
				 classifier



scenedetect.py

	This files runs the functions in utils.py file to create the .npy files. Since 
	the runtime of making vocabulary and Bow is long, we decided to use the .npy file 
	instead of keep making new vocabularies and Bows. Once the .npy files for 
	vocabulary and bow are created, we load the files to use as our vocabulary and 
	bow to check the accuracy and runtime. 




Reference

	[1] KNeighborsClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

	[2] SVC - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

	[3] CV2 Normalization - https://stackoverflow.com/questions/40645985/opencv-python-normalize-image/42164670

	[4] K Means - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

	[5] AgglomerativeClustering - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

	[6] Sift, Surf, Orb - https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/