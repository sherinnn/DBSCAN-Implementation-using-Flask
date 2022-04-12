from flask import request
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

  
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def dbscan(eps, min_samples, path):
    print("hello")
    # X = pd.read_csv('uploads/cereal.csv')
    # db=DBSCAN(eps=0.3,min_samples=5,metric='euclidean')
#     model=db.fit(X)
#     label=model.labels_
#     sample_cores=np.zeros_like(label,dtype=bool)

#     sample_cores[db.core_sample_indices_]=True

# #Calculating the number of clusters

#     n_clusters=len(set(label))- (1 if -1 in label else 0)
#     print('No of clusters:',n_clusters)



# Scaling the data to bring all the attributes to a comparable level
    X = pd.read_csv(path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("hello")

    # Normalizing the data so that
    # the data approximately follows a Gaussian distribution
    X_normalized = normalize(X_scaled)

    # Converting the numpy array into a pandas DataFrame
    X_normalized = pd.DataFrame(X_normalized)
    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    print(X_principal.head())
    # Numpy array of all the cluster labels assigned to each data point
    db_default = DBSCAN(eps =eps, min_samples = min_samples).fit(X_principal)
    labels = db_default.labels_
    print(labels)    # Building the label to colour mapping
    colours = {}
    colours[0] = 'r'
    colours[1] = 'g'
    colours[2] = 'b'
    colours[-1] = 'k'

    # Building the colour vector for each data point
    cvec = [colours[label] for label in labels]

    # For the construction of the legend of the plot
    r = plt.scatter(X_principal['P1'], X_principal['P2'], color ='r');
    g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g');
    b = plt.scatter(X_principal['P1'], X_principal['P2'], color ='b');
    k = plt.scatter(X_principal['P1'], X_principal['P2'], color ='k');

    # Plotting P1 on the X-Axis and P2 on the Y-Axis
    # according to the colour vector defined
    plt.figure(figsize =(9, 9))
    plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)

    # Building the legend
    plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))

    db = DBSCAN(eps = 0.375, min_samples = 50).fit(X_principal)
    labels1 = db.labels_
    colours1 = {}
    colours1[0] = 'r'
    colours1[1] = 'g'
    colours1[2] = 'b'
    colours1[3] = 'c'
    colours1[4] = 'y'
    colours1[5] = 'm'
    colours1[-1] = 'k'

    cvec = [colours1[label] for label in labels]
    colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ]

    r = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[0])
    g = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[1])
    b = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[2])
    c = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[3])
    y = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[4])
    m = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[5])
    k = plt.scatter(
            X_principal['P1'], X_principal['P2'], marker ='o', color = colors[6])

    no_clusters = len(np.unique(labels) )
    print('Estimated no. of clusters: %d' % no_clusters)
    plt.figure(figsize =(9, 9))
    plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)
    print("hello")
    plt.xlabel("P1")
    plt.ylabel("P2")
    plt.savefig('static/my_plot.png')


    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

    # X = StandardScaler().fit_transform(X)

    # # Compute DBSCAN
    # db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #     % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #     % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #     % metrics.silhouette_score(X, labels))

    # # Plot result


    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]

    #     class_member_mask = (labels == k)

    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #             markeredgecolor='k', markersize=14)

    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #             markeredgecolor='k', markersize=6)

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.savefig('my_plot.png')
    
    


    # db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each)
    #     for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #     # Black used for noise.
    #         col = [0, 0, 0, 1]

    # class_member_mask = (labels == k)

    # xy = X[class_member_mask & core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=14)

    # xy = X[class_member_mask & ~core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=6)

    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.savefig('my_plot.png')





