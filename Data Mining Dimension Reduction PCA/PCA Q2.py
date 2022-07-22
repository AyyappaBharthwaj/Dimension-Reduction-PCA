# -*- coding: utf-8 -*-
"""

"""
################# PCA(PRINCIPAL COMPONENT ANALYSIS) ############

"""A pharmaceuticals manufacturing company is conducting a study on a new medicine to treat
 heart diseases. The company has gathered data from its secondary sources and would like you
to provide high level analytical insights on the data. Its aim is to segregate patients 
depending on their age group and other factors given in the data. Perform PCA and clustering
algorithms on the dataset and check if the clusters formed before and after PCA are the same
and provide a brief report on your model. You can also explore more ways to improve your model. 
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


Heart_disease=pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\heart disease.csv")
Heart_disease.info()#dtypes: float64(11), int64(3) and memory usage: 19.6 KB
Heart_disease
### data preprocesing  ##############
Heart_disease.isna().sum()# no null values
Heart_disease.duplicated().sum()# 1 duplicate value

df1 = Heart_disease.drop_duplicates()

Heart_disease.columns

#type casting
Heart_disease.oldpeak=Heart_disease.oldpeak.astype("int64") #changing float data type to int datatype
Heart_disease.dtypes


#finding outliers
sns.boxplot(Heart_disease["age"])
sns.boxplot(Heart_disease["sex"])
sns.boxplot(Heart_disease["cp"])
sns.boxplot(Heart_disease["trestbps"]) #outliers are present
sns.boxplot(Heart_disease["chol"])#outliers are present
sns.boxplot(Heart_disease["fbs"])#outliers are present
sns.boxplot(Heart_disease["restecg"])
sns.boxplot(Heart_disease["thalach"]) #outliers are present
sns.boxplot(Heart_disease["exang"])
sns.boxplot(Heart_disease["oldpeak"]) # outliers are present
sns.boxplot(Heart_disease["slope"])
sns.boxplot(Heart_disease["ca"]) #outliers are present
sns.boxplot(Heart_disease["thal"]) # outliers are present
sns.boxplot(Heart_disease["target"])
#outliers are present in 7 columns

# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['trestbps'].quantile(0.75) - Heart_disease['trestbps'].quantile(0.25)
lower_limit = Heart_disease['trestbps'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['trestbps'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['trestbps'])

df_t = winsor.fit_transform(Heart_disease[['trestbps']])

# lets see boxplot
sns.boxplot(df_t.trestbps) # no outliers in trestbps data


# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['chol'].quantile(0.75) - Heart_disease['chol'].quantile(0.25)
lower_limit = Heart_disease['chol'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['chol'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['chol'])

df_t = winsor.fit_transform(Heart_disease[['chol']])

# lets see boxplot
sns.boxplot(df_t.chol) # no outliers in chol data


# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['fbs'].quantile(0.75) - Heart_disease['fbs'].quantile(0.25)
lower_limit = Heart_disease['fbs'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['fbs'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['fbs'])

df_t = winsor.fit_transform(Heart_disease[['fbs']])

# lets see boxplot
sns.boxplot(df_t.fbs) # no outliers in fbs data

# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['thalach'].quantile(0.75) - Heart_disease['thalach'].quantile(0.25)
lower_limit = Heart_disease['thalach'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['thalach'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['thalach'])

df_t = winsor.fit_transform(Heart_disease[['thalach']])

# lets see boxplot
sns.boxplot(df_t.thalach) # no outliers in thalach data


# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['oldpeak'].quantile(0.75) - Heart_disease['oldpeak'].quantile(0.25)
lower_limit = Heart_disease['oldpeak'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['oldpeak'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['oldpeak'])

df_t = winsor.fit_transform(Heart_disease[['oldpeak']])

# lets see boxplot
sns.boxplot(df_t.oldpeak) # no outliers in oldpeak data



# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['ca'].quantile(0.75) - Heart_disease['ca'].quantile(0.25)
lower_limit = Heart_disease['ca'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['ca'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['ca'])

df_t = winsor.fit_transform(Heart_disease[['ca']])

# lets see boxplot
sns.boxplot(df_t.ca) # no outliers in ca data



# Detection of outliers (find limits based on IQR)
IQR = Heart_disease['thal'].quantile(0.75) - Heart_disease['thal'].quantile(0.25)
lower_limit = Heart_disease['thal'].quantile(0.25) - (IQR * 1.5)
upper_limit = Heart_disease['thal'].quantile(0.75) + (IQR * 1.5)

####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['thal'])

df_t = winsor.fit_transform(Heart_disease[['thal']])

# lets see boxplot
sns.boxplot(df_t.thal) # no outliers in thal data


#normalization
# converts range to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

Heart_norm = norm_func(Heart_disease)
# now our data is scale free and unit free


############### Exploratory Data Analysis##################
# Measures of Central Tendency / First moment business decision
Heart_disease.mean() #high mean in chol column
Heart_disease.median()#high median in chol column
#mean>median

from scipy import stats
stats.mode(Heart_disease)

# Measures of Dispersion / Second moment business decision
Heart_disease.var() # high variance in chol column
#high variance means more data spread 
Heart_disease.std() # high standard deviation in chol column

# Third moment business decision
Heart_disease.skew()

# Fourth moment business decision
Heart_disease.kurt()

############## data visuvalization #############
#histogram
plt.hist(Heart_disease,bins=9)#data is moderatly distributed normally


############### hierarchical clustering #############3

######### creating dendrogram  ######### 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

#computing the distance between data points using euclidean distance
a= linkage(Heart_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(25, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(a, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()
#by dendogram we can say that more distance of vertical lines means more distance b/w those clusters
#initialy it form 10 diff color culters.i.e 10 different groups


from sklearn.cluster import AgglomerativeClustering
# no of cluster depends on clint's requirements as it is subjective.
#we can change that as per business requirements 
#if we draw a horizantal at distance 1.5 that line joint 6 horizantal lines
#so we can take no of cluster as 10

a_complete = AgglomerativeClustering(n_clusters = 10, linkage = 'complete', affinity = "euclidean").fit(Heart_norm) 
a_complete.labels_ #it gives array of 6clsters 
cluster_labels = pd.Series(a_complete.labels_)

Heart_disease['clust'] = cluster_labels # creating a new column and assigning it to new column 
Heart_disease.insert(0,"clust",Heart_disease.pop("clust"))#it give 0 indexing to clust column

Heart= Heart_disease.iloc[:, 0:15]
Heart.head()

# Aggregate mean of each cluster
a1=Heart_disease.iloc[:, :].groupby(Heart_disease.clust).mean()

#clust0 contains high oldpeak,ca, and has more age
#clust4 contains high fbs
#clust8 contains high trestbps and restecg
# clust7 contains high cp and thalach
#clust9 contains high chol


#################   k-means #################
from sklearn.cluster import	KMeans 

###### scree plot or elbow curve ############
TWSS = [] #creating empty list for total within sum of sqr
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Heart_norm)
    TWSS.append(kmeans.inertia_)#inertia is inbuild fn which gives twss
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
#from 2 to 3 and 3 to 4 it covered more data change and 4 to 5  little less than 2 to 3
#so we can consider no of k mean clusters as 4

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(Heart_norm)

model.labels_ # getting the labels of clusters assigned to each row 
k=pd.Series(model.labels_)  # converting numpy array into pandas series object 
Heart_disease['clust']=k # creating a  new column and assigning it to new column 

Heart_disease.head()

Heart_disease = Heart_disease.iloc[:,:]
Heart_disease.head()

x=Heart_disease.iloc[:, :].groupby(Heart_disease.clust).mean()

# clust0 contains high chol
# clust1 contains more ca and has highest age
# clust2 contains more trestbps, excang, oldpeak and thal
#clust3 contains more cp,fbs,restecg,thalach,slope,

###################### PCA ####################

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(Heart_norm)

# The amount of variance that each PCA  
var = pca.explained_variance_ratio_
var # variance of each column  #array([0.33065242, 0.15318074, 0.10547242, 0.09844422, 0.06958699, 0.05162349])

# PCA weights
pca.components_
pca.components_[0] #pc1 values

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
#here we r taking var & rounding it upto 4 decimals and multiply with 100 for %
var1  # array([33.07, 48.39, 58.94, 68.78, 75.74, 80.9 ])

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns='pc0','pc1','pc2','pc3','pc4','pc5'
final_pca = pd.concat([Heart_disease, pca_data.iloc[:, 0:]], axis = 1)

# Scatter diagram
plot = final_pca.plot(x='pc0', y='pc1', kind='scatter',figsize=(12,8))

#now new dataset  is Heartdisease_pca
#let's perform hierarchical and K-means clustering on Heartdisease_pca


############### hierarchical clustering #############3

######### creating dendrogram  ######### 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

#computing the distance between data points using euclidean distance
b= linkage(final_pca, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(25, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(b, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size =4 # font size for the x axis labels
)
plt.show()
#we are able to see clear visualization after pca

from sklearn.cluster import AgglomerativeClustering

b_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(Heart_norm) 
b_complete.labels_  
cluster_labels = pd.Series(b_complete.labels_)

final_pca['clust'] = cluster_labels # creating a new column and assigning it to new column 
final_pca.insert(0,"clust",final_pca.pop("clust"))#it give 0 indexing to clust column

final1_pca = final_pca.iloc[:, 0:]
final1_pca.head()

# Aggregate mean of each cluster
b1=final_pca.iloc[:, :].groupby(final_pca.clust).mean()
#clust1 of pc0 has more mean
#clust1 has  has more mean so it is prefered  heartdisease

#################   k-means #################
from sklearn.cluster import	KMeans 

###### scree plot or elbow curve ############
TWSS = [] #creating empty list for total within sum of sqr
k1 = list(range(2, 9))

for i in k1:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final_pca)
    TWSS.append(kmeans.inertia_)#inertia is inbuild fn which gives twss
    
TWSS
# Scree plot 
plt.plot(k1, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
#from 2 to 3 amd 3 to 4 it covers max datapoints
#when compare to before pca k-means after pca k-meanscovers max datapoints
#at 2 to 3 and 3 to 4

# we can take no of cluster=3 from the above twss values
model1 = KMeans(n_clusters = 4)
model1.fit(final_pca)

model1.labels_ # getting the labels of clusters assigned to each row 
k1=pd.Series(model1.labels_)  # converting numpy array into pandas series object 
final_pca['clust']=k1 # creating a  new column and assigning it to new column 
final_pca.head()

final_pca = final_pca.iloc[:,:]
final_pca.head()

y=final_pca.iloc[:, :].groupby(final_pca.clust).mean()
y
 
#clust2 has more mean and  so it is most preffered one  
#we get similar results
#but pca is not as readable and interpretable because we lose some features















