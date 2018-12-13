from recommendation_system import model_select
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
import pandas as pd
from surprise import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import surprise as sp
from surprise import Reader
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")




def file_IO():
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

    d1 = pd.read_csv("~/Desktop/Tufts/Fall2018/COMP135/Project3/trainset.csv")

    d2 = pd.read_csv("~/Desktop/Tufts/Fall2018/COMP135/Project3/testset.csv")

    data = pd.concat((d1,d2))
    data2 = Dataset.load_from_df(data[["user", "item", "rating"]], reader=reader)

    train = data2.build_full_trainset()

    gender = np.genfromtxt('gender.csv', delimiter=',')
    gender = gender[1:]
    gender  = gender.reshape(-1, 1)


    years = np.genfromtxt('release-year.csv', delimiter=',')
    years = years[1:]
    years = years.reshape(-1,1)

    return train, gender, years

"""# DRIVER - Uncomment to run

train, gender, years = file_IO()

# Gender model using Logistic Regression 
clf = sp.SVD()
clf.fit(train)
X = clf.pu
X = pd.DataFrame.from_records(X)
clf2 = LogisticRegression(C= .01,  solver='lbfgs')
clf2.fit(X, gender)

pred = -1 * cross_val_score(clf2, X, gender, scoring=("neg_mean_squared_error"), cv=10)
print "\nMSE of model trained on user features:", pred

# Release year model using Support Vector Regression
V = clf.qi
V = pd.DataFrame.from_records(V)
clf3 = SVR(C=.01, epsilon=.01)
clf3.fit(V, years)


scores = cross_val_score(clf3, V, years, scoring=("neg_mean_squared_error"), cv=10)
#print cross_validate(clf3, V, years, scoring=("neg_mean_squared_error"))
print "\nSVR: MSE of model trained on movie features:",  -1 * scores

# Naive model for release years
mean_year = np.mean(years)
clf4 = SVR(C=.01, epsilon=.01)
mean_yr_ray = np.full((len(years), 1), mean_year)
clf4.fit(mean_yr_ray, years)
pred = clf4.predict(mean_yr_ray)

print "\nNaive SVR for mean release year:", metrics.mean_squared_error(years, pred)


V = clf.qi
V = pd.DataFrame.from_records(V)
clf3 = LinearRegression(fit_intercept=True, normalize=True)
clf3.fit(V, years)


scores = cross_val_score(clf3, V, years, scoring=("neg_mean_squared_error"), cv=10)
#cross_validate(clf3, V, years, scoring=("neg_mean_squared_error"))

print "\nLinReg: MSE of model trained on movie features:",  -1 * scores


# Naive model for release years
mean_year = np.mean(years)
clf4 = LinearRegression(fit_intercept=True, normalize=True)
mean_yr_ray = np.full((len(years), 1), mean_year)
clf4.fit(mean_yr_ray, years)
pred = clf4.predict(mean_yr_ray)
print "\nNaive LinReg for mean release year:", metrics.mean_squared_error(years, pred)

"""