# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 05:51:14 2018

@author: I870266
"""

import pandas as pd
import numpy as np                # for mathematics
import matplotlib.pyplot as plt   # for nice charts and graphs

test_dataset = pd.read_csv('adding_classif_n_templ_to_pre_clus.csv')

without_z = test_dataset[(test_dataset["group"] != "Z") & (test_dataset["percent_budget_change"] <= 100)]
x = without_z.iloc[:, [29, 30, 31, 32] ].values # contains the "days_extended" & the "budget_percent_change" columns

#visualizing our data using a matrix of scatterplots a scatterplot
from pandas.plotting import scatter_matrix
scatter_matrix(without_z.iloc[:, [29, 30, 31, 32] ])
plt.show()


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_sc = sc.fit_transform(x)


#Applying Kernal CPA for Feature extraction cuz the two largest clusters (classes) in our dataset are not linearly separable
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 4, kernel = "rbf")
x_sc_pca = kpca.fit_transform(x_sc)






#Using the "Elbow method" to find the optimal number of clusters
from sklearn.cluster import KMeans

#the "WITHIN CLUSTER SUM OF SQUARES" (WCSS) Metric
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10) # we are setting the "k-means++" vakue to the "init" parameter to avoid falling into the initial randomization trap
                                                                                                       # max_iter default value is 300
                                                                                                       # n_init default value is 10
                                                                                                       # random state can take a "random" parameter or any integer that would lead to creating a unique seed. If i choose "7", anyone using 7 would get the same result.
                                                                                                         #(In a way, it saves the random result into a seed number)
    #let's fit the "KMeans" algorithm to our set "x"
    kmeans.fit(x_sc_pca)
    wcss.append(kmeans.inertia_)   #the sci-kit libraries has a built in algorithm for computing the wcss which is called "inertia_"

#Plot the "Elbow Method" Graph
plt.plot(range(1, 11), wcss)    # range(1, 11) is a smart way of laying out the numbers in that range as x-axis values
                                # wcss is the y-axis
plt.title("Elbow Method Graph(K Means)")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
Elbow_Method_Graph_K_Means = plt.show()    # to display the graph

#Applying k-means to our dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x_sc_pca)

without_z["clusters"] = y_kmeans

without_z.to_csv("clustered using kernel.csv")
