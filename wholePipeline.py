import argparse
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def getFiles(path):
    images = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))
    return images


def getDescriptors(descMethod, img):
    kp, des = descMethod.detectAndCompute(img, None)
    return des


def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))


def vstackDescriptors(descriptor_list):
    #print('The shape of descriptor list is ')
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        #print('Descriptor is')
        #print(descriptor)
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(len(descriptor_list))])
    for i in range(len(descriptor_list)):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features


def normalizeFeatures(scale, features):
    return scale.transform(features)


def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel):
    features = im_features
    if (kernel == "precomputed"):
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param)
    svm.fit(features, train_labels)
    return svm


def plotConfusionMatrix(y_true, y_pred, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plotConfusions(true, predictions, dict):
    np.set_printoptions(precision=2)

    class_names = dict.keys()
    plotConfusionMatrix(true, predictions, classes=class_names,
                        title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()


def findAccuracy(true, predictions):
    print('accuracy score: %0.3f' % accuracy_score(true, predictions))


def trainModel(path, no_clusters, kernel, dict):
    images = getFiles(path)
    print("Train images path detected.")
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    image_count = len(images)


    for img_path in images:
        class_name = os.path.basename(os.path.dirname(img_path))
        class_index = dict[class_name]
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        if(des is not None):
            descriptor_list.append(des)
            train_labels = np.append(train_labels, class_index)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    plotHistogram(im_features, no_clusters)
    print("Features histogram plotted.")

    svm = findSVM(im_features, train_labels, kernel)
    print("SVM fitted.")
    print("Training completed.")

    return kmeans, scale, svm, im_features


def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel, dict):
    test_images = getFiles(path)
    print("Test images path detected.")

    count = 0
    true = []
    descriptor_list = []

    sift = cv2.xfeatures2d.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)

        if (des is not None):
            count += 1
            descriptor_list.append(des)
            class_name = os.path.basename(os.path.dirname(img_path))
            true.append(class_name)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features
    if (kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)

    r_dict = {v:k for k, v in dict.items()}
    predictions = [r_dict[i] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    plotConfusions(true, predictions, dict)
    print("Confusion matrixes plotted.")

    findAccuracy(true, predictions)
    print("Accuracy calculated.")
    print("Execution done.")


def execute(train_path, test_path, no_clusters, kernel):
    d = os.listdir(train_path)
    dict = {d[i]:i for i in range(len(d))}
    kmeans, scale, svm, im_features = trainModel(train_path, no_clusters, kernel, dict)
    testModel(test_path, kmeans, scale, svm, im_features, no_clusters, kernel, dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)
    parser.add_argument('--no_clusters', action="store", dest="no_clusters", default=50)
    parser.add_argument('--kernel_type', action="store", dest="kernel_type", default="linear")

    args = vars(parser.parse_args())
    if (not (args['kernel_type'] == "linear" or args['kernel_type'] == "precomputed")):
        print("Kernel type must be either linear or precomputed")
        exit(0)

    execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'])
