from utils import *
import argparse

parser = argparse.ArgumentParser(description='arg options')
parser.add_argument("--tiny", "-t", type=bool, default=True, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
args = parser.parse_args()



if __name__ == "__main__":
    
    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results') 
        SAVEPATH = 'Results/'
    
    # Load data
    train_images, test_images, train_labels, test_labels = load_data()
    
    if args.tiny:
        tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
    
        # Split accuracies and runtimes for saving  
        for element in tinyRes[::2]:
            # Check that every second element is an accuracy in reasonable bounds
            assert (7 < element and element < 25)
        acc = np.asarray(tinyRes[::2])
        runtime = np.asarray(tinyRes[1::2])
    
        # Save results
        np.save(SAVEPATH + 'tiny_acc.npy', acc)
        np.save(SAVEPATH + 'tiny_time.npy', runtime)

    # # Create vocabularies, and save them in the result directory
    vocabularies = []
    vocab_idx = []
    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                vocabulary = buildDict(train_images, dict_size, feature, algo)
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename, np.asarray(vocabulary))
                vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary
                
    # # Compute the Bow representation for the training and testing sets
    test_rep = [] # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = [] # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift', 'surf', 'orb'] # Order in which features were used 
    # for vocabulary generation

    train_rep = []
    test_rep = []
    count = 1
    for f in range(len(features)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                for image in train_images:
                    rep = computeBow(image, np.load(SAVEPATH + 'voc_'+features[f]+'_'+algo+'_'+str(dict_size)+'.npy'), features[f])
                    train_rep.append(rep)
                for t_image in test_images:
                    t_rep = computeBow(t_image, np.load(SAVEPATH + 'voc_'+features[f]+'_'+algo+'_'+str(dict_size)+'.npy'), features[f])
                    test_rep.append(t_rep)
                np.save(SAVEPATH + 'bow_test_'+ str(count) + '.npy', np.asarray(test_rep))
                np.save(SAVEPATH + 'bow_train_'+ str(count) + '.npy', np.asarray(train_rep))
                train_rep = []
                test_rep = []
                count += 1
        
    
    # Use BOW features to classify the images with a KNN classifier
    # A list to store the accuracies and one for runtimes
    knn_accuracies = []
    knn_runtimes = []


    count = 0
    for f in range(len(features)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                train_rep = np.load(SAVEPATH + 'bow_train_' + str(count) + '.npy')
                test_rep = np.load(SAVEPATH + 'bow_test_' + str(count) + '.npy')
                t1 = time.time()
                k = KNN_classifier(train_rep, train_labels, test_rep, 9)
                knn_accuracies.append(reportAccuracy(test_labels, k))
                knn_runtimes.append(time.time()-t1)
                count += 1
          
    np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory

    
    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    

    count = 0
    for f in range(len(features)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                train_rep = np.load(SAVEPATH + 'bow_train_' + str(count) + '.npy')
                test_rep = np.load(SAVEPATH + 'bow_test_' + str(count) + '.npy')
                t1 = time.time()
                k = SVM_classifier(train_rep, train_labels, test_rep, True, 10)
                lin_accuracies.append(reportAccuracy(test_labels, k))
                lin_runtimes.append(time.time()-t1)
                count += 1
                
    np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory
    

    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []
    

    count = 0
    for f in range(len(features)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                train_rep = np.load(SAVEPATH + 'bow_train_' + str(count) + '.npy')
                test_rep = np.load(SAVEPATH + 'bow_test_' + str(count) + '.npy')
                t1 = time.time()
                k = SVM_classifier(train_rep, train_labels, test_rep, False, 10)
                rbf_accuracies.append(reportAccuracy(test_labels, k))
                rbf_runtimes.append(time.time()-t1)
                count += 1
                
    np.save(SAVEPATH +'rbf_accuracies.npy', np.asarray(rbf_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH +'rbf_runtimes.npy', np.asarray(rbf_runtimes)) # Save the runtimes in the Results/ directory
            
    
