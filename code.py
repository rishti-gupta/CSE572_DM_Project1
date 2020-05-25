#!/usr/bin/env python
# coding: utf-8

# In[19]:


# importing python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create expanding window features
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import warnings
warnings.filterwarnings('ignore')

# loading the csv files
df_timestamps1 = pd.read_csv('Datafolder//CGMDatenumLunchPat1.csv')
df_glucoseLevels1 = pd.read_csv('DataFolder//CGMSeriesLunchPat1.csv')
df_timestamps2 = pd.read_csv('DataFolder//CGMDatenumLunchPat2.csv')
df_glucoseLevels2 = pd.read_csv('DataFolder//CGMSeriesLunchPat2.csv')
df_timestamps3 = pd.read_csv('DataFolder//CGMDatenumLunchPat3.csv')
df_glucoseLevels3 = pd.read_csv('DataFolder//CGMSeriesLunchPat3.csv')
df_timestamps4 = pd.read_csv('DataFolder//CGMDatenumLunchPat4.csv')
df_glucoseLevels4 = pd.read_csv('DataFolder//CGMSeriesLunchPat4.csv')
df_timestamps5 = pd.read_csv('DataFolder//CGMDatenumLunchPat5.csv')
df_glucoseLevels5 = pd.read_csv('DataFolder//CGMSeriesLunchPat5.csv')

# concatinating the whole dataset
df_glucoseLevels = pd.concat([df_glucoseLevels1, df_glucoseLevels2, df_glucoseLevels3,df_glucoseLevels4, df_glucoseLevels5])
df_timestamps = pd.concat([df_timestamps1, df_timestamps2, df_timestamps3,df_timestamps4, df_timestamps5])

# create a dataframe
df_features= pd.DataFrame()

# df_timestamps = df_timestamps.rename(columns={'cgmDatenum_ 9': 'cgmDatenum_9', 'cgmDatenum_ 8': 'cgmDatenum_8',
#                                               'cgmDatenum_ 7': 'cgmDatenum_7', 'cgmDatenum_ 6': 'cgmDatenum_6',
#                                               'cgmDatenum_ 5': 'cgmDatenum_5', 'cgmDatenum_ 4': 'cgmDatenum_4',
#                                               'cgmDatenum_ 3': 'cgmDatenum_3', 'cgmDatenum_ 2': 'cgmDatenum_2',
#                                               'cgmDatenum_ 1': 'cgmDatenum_1'})

# replace all missing values with zeros
df_glucoseLevels.fillna(0, inplace=True)
df_timestamps.fillna(0, inplace=True)

# calculating feature-1: rolling mean and standard deviation
rolling_mean = df_glucoseLevels.rolling(window=3,min_periods=3).mean()
rolling_std = df_glucoseLevels.rolling(window=3,min_periods=3).std()

# rolling_mean
# rolling_std

# plotting feature-1 
# rolling mean plot
import matplotlib.pyplot as plt

df_timestamps_arr = np.array(df_timestamps)
df_glucoseLevels_arr = np.array(df_glucoseLevels)
rolling_mean_arr = np.array(rolling_mean)
x = [i for i in range(len(df_timestamps_arr[32]))]
plt.plot(x, rolling_mean_arr[32])
plt.title("CGM:Rolling Mean")
plt.xlabel("Time Series")
plt.ylabel("Rolling Mean")
plt.show()
#stem plot of rolling mean
plt.stem(x, rolling_mean_arr[32])
plt.title("CGM:Rolling Mean")
plt.xlabel("Time Series")
plt.ylabel("Rolling Mean")
plt.show()

# rolling standard deviation plot
import matplotlib.pyplot as plt

rolling_std = np.array(rolling_std)
x = [i for i in range(len(df_timestamps_arr[30]))]
plt.plot(x, rolling_std[30])
plt.title("CGM:Rolling Standard Deviation")
plt.xlabel("Time Series")
plt.ylabel("Rolling Standard Deviation")
plt.show()
# stem plot of rolling standard deviation
plt.stem(x, rolling_std[30])
plt.title("CGM:Rolling Standard Deviation")
plt.xlabel("Time Series")
plt.ylabel("Rolling Standard Deviation")
plt.show()

# calculating feature-2: FFT
import scipy.fftpack

cgmFFTValues = abs(np.fft.fft(df_glucoseLevels_arr))
# print("FFT Values = ", cgmFFTValues)
freq = np.fft.fftfreq(df_timestamps_arr.shape[-1])
# print("FFT Frequencies = ", freq)
plt.stem(freq,cgmFFTValues[10])

plt.title("Fast Fourier Transform")
plt.ylabel("FFT Values")
plt.xlabel("frequency")
plt.show()

# pickign top 8 peaks of FFT
FFT=np.array(cgmFFTValues)
fft_freq=np.array(freq)
Fourier_peak=list()
Fourier_frequency=list()
for i in range(len(FFT)):
    index=np.argsort(FFT)[i][-9:]

    peak=FFT[i][index]
    Fourier_peak.append(peak)
    freq=abs(fft_freq[index])
    freq.sort()
    fr=freq[[0,1,3,5,7]]
    Fourier_frequency.append(fr)

Fourier_peak=np.array(Fourier_peak)
Fourier_frequency=np.array(Fourier_frequency)
Fourier_peak=np.unique(Fourier_peak,axis=1)

# Extracting feature-3: polyfit regression

polyfit_reg = []
x = [i for i in range(len(df_timestamps_arr[32]))]

for i in range(len(df_glucoseLevels_arr)):
    polyfit_reg.append(np.polyfit(x, df_glucoseLevels_arr[i], 3))

polyfit_reg = np.array(polyfit_reg)
# polyfit_reg plot (record no. 10)
plt.plot(x, np.polyval(polyfit_reg[10], x), label='polyfit')
plt.plot(x, df_glucoseLevels_arr[10], label='actual data')
plt.legend()
plt.ylabel("CGM Series")
plt.xlabel("Time Series")
plt.show()

# Extracting feature-4: Interquartile Range
IQR = []
for i in range(len(df_glucoseLevels)):
    Q1 = np.percentile(df_glucoseLevels_arr[i], 25, interpolation = 'midpoint')
    Q3 = np.percentile(df_glucoseLevels_arr[i], 75, interpolation = 'midpoint') 
    IQR.append(Q3 - Q1) 

IQR = np.array(IQR)

x=list()
x.append(IQR)
x=np.array(x)
x = x.T

# creating the feature matrix by appending all the features extracted
feature_matrix = np.append(rolling_mean, rolling_std, axis = 1)
feature_matrix = np.append(feature_matrix, Fourier_frequency, axis = 1)
feature_matrix = np.append(feature_matrix, Fourier_peak, axis = 1)
feature_matrix = np.append(feature_matrix, polyfit_reg, axis = 1)
feature_matrix = np.append(feature_matrix, x, axis = 1)

# Tackling the NAN values/missing values by replacing them with zeros
from numpy import *
where_are_NaNs = isnan(feature_matrix)
feature_matrix[where_are_NaNs] = 0
# print("Feature Matrix = ", feature_matrix)

# create the covariance matrix
import numpy as np
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
X_std = sc.fit_transform(feature_matrix)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
# print('Covariance matrix = \n%s' %cov_mat)

# create eigen values and eigen vectors
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# feed the feature matrix to PCA

from sklearn.preprocessing import StandardScaler

feature_matrix = StandardScaler().fit_transform(feature_matrix)
df_feature_matrix = pd.DataFrame(feature_matrix)
df_feature_matrix.fillna(0, inplace=True)

from sklearn import preprocessing
data_scaled = pd.DataFrame(preprocessing.scale(df_feature_matrix), columns=df_feature_matrix.columns)
pca = decomposition.PCA(n_components=5)
pca1 = decomposition.PCA(n_components=30)
X_std_pca = pca.fit_transform(data_scaled)
X_std_pca1 = pca1.fit_transform(data_scaled)
# X_std_pca.shape

# calculate the explained variance of pca
pcaExpVariance = pca.explained_variance_
# print("PCA variance= ", pcaExpVariance)
pcaTransformed = pca.transform(feature_matrix)
# pcaTransformed.shape

# calculate explained variance ratio for analysis of no. of features
# using 5 components
variance = pca.explained_variance_ratio_
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
print("Explained Variance Ratio (5 components) = ", var)

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis (5 components)')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)
plt.show()

# using 30 components
variance1 = pca1.explained_variance_ratio_
var1=np.cumsum(np.round(pca1.explained_variance_ratio_, decimals=3)*100)
print("Explained Variance Ratio (30 components) = ", var1)

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis (30 components)')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var1)
plt.show()


# plotting all the variances of features captured by pca
pca_Data_Frame = pd.DataFrame(pca.components_, columns=data_scaled.columns,
                              index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5'])
pca_Data_Frame.to_csv('PCA_FEATURES_MATRIX.csv')
# print(pca_Data_Frame)

pca_Data_Frame.idxmin(axis=1)

plt.scatter(range(len(df_glucoseLevels_arr)), X_std_pca[:, 0], marker='o', color='b', label='First PCA');
plt.scatter(range(len(df_glucoseLevels_arr)), X_std_pca[:, 1], marker='o', color='y', label='Second PCA');
plt.scatter(range(len(df_glucoseLevels_arr)), X_std_pca[:, 2], marker='o', color='g', label='Third PCA');
plt.scatter(range(len(df_glucoseLevels_arr)), X_std_pca[:, 3], marker='o', color='r', label='Fourth PCA');
plt.scatter(range(len(df_glucoseLevels_arr)), X_std_pca[:, 4], marker='o', color='c', label='Fifth PCA');

plt.legend()
plt.xlabel('Time Series')
plt.ylabel('Variances')

plt.show()

# scree/bar plot of variances
plt.bar(list(range(0, 5)), pcaExpVariance)

plt.xlabel('Principal Components')
plt.ylabel('Variances')
plt.show()

# plotting individual pca components with time

plt.scatter(list(range(0, 216)), pcaTransformed[0:216, 0], color='b')
plt.title("PCA-1")
plt.scatter(range(len(df_glucoseLevels_arr)), pcaTransformed[:, 0], marker='o', color='b', label='First PCA');
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.ylim(-3,3)
plt.legend()
plt.show()

plt.title("PCA-2")
plt.scatter(range(len(df_glucoseLevels_arr)), pcaTransformed[:, 1], marker='o', color='y', label='Second PCA');
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.legend()
plt.show()

plt.title("PCA-3")
plt.scatter(range(len(df_glucoseLevels_arr)), pcaTransformed[:, 2], marker='o', color='g', label='Third PCA');
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.legend()
plt.show()

plt.title("PCA-4")
plt.scatter(range(len(df_glucoseLevels_arr)), pcaTransformed[:, 3], marker='o', color='r', label='Fourth PCA');
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.legend()
plt.show()

plt.title("PCA-5")
plt.scatter(range(len(df_glucoseLevels_arr)), pcaTransformed[:, 4], marker='o', color='c', label='Fifth PCA');
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.legend()
plt.show()


# In[6]:


# printing the results of all the features extracted

print("Rolling Mean = ", rolling_mean)
print("rolling Standard Deviation = ", rolling_std)

print("---------------------------------------------------------------------------------")

Fourier_peak = pd.DataFrame(Fourier_peak)
print("Fourier Values = ", Fourier_peak)

print("---------------------------------------------------------------------------------")

Fourier_frequency = pd.DataFrame(Fourier_frequency)
print("Fourier Frequency = ", Fourier_frequency)

print("---------------------------------------------------------------------------------")


polyfit_reg = pd.DataFrame(polyfit_reg)
print("Polynomial fit coefficients (3-degree) = ",polyfit_reg)

print("---------------------------------------------------------------------------------")

x = pd.DataFrame(x)
print("IQR = ",x)

print("---------------------------------------------------------------------------------")

feature_matrix = pd.DataFrame(feature_matrix)
print("Final Feature Matrix = ",feature_matrix)

print("---------------------------------------------------------------------------------")

cov_mat = pd.DataFrame(cov_mat)
print("Covariance Matrix = ",cov_mat)

print("---------------------------------------------------------------------------------")

eig_vals = pd.DataFrame(eig_vals)
print("Eigen Values = ",eig_vals)

print("---------------------------------------------------------------------------------")

eig_vecs = pd.DataFrame(eig_vecs)
print("Eigen Vectors= ", eig_vecs)

print("---------------------------------------------------------------------------------")

X_std_pca = pd.DataFrame(X_std_pca)
print("PCA Matrix = ", X_std_pca)

print("---------------------------------------------------------------------------------")

pcaExpVariance = pd.DataFrame(pcaExpVariance)
print("PCA Explained Variance", pcaExpVariance)

print("---------------------------------------------------------------------------------")
      


# In[ ]:




