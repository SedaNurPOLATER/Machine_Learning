# sklearn datasetlerinden olan iris datasetini import edelim
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%
iris = load_iris()

x = iris.data
y = iris.target

# %% normalization kullanıyoruz çünkü knn algoritması kullanacağız
x = (x-np.min(x))/(np.max(x)-np.min(x))

# %% train test split olarak data'yı 2 ye ayırıyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3) # %30 test olsun

#%% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)  # k = n_neighbors 

# %% K fold CV K = 10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = x_train, y= y_train, cv = 10)
print("average accuracy: ",np.mean(accuracies)) # ortalama aldık
print("average std: ",np.std(accuracies)) # standart sapmasına bakalım, yani yayılıma bakıyoruz

#%% Burada artık test verilerini kullanıyoruz
knn.fit(x_train,y_train)
print("test accuracy: ",knn.score(x_test,y_test))

# %% grid search cross validation for knn

from sklearn.model_selection import GridSearchCV
# grid belirlemek her kombinasyon için bir modeli değerlendirir
grid = {"n_neighbors":np.arange(1,50)}
knn= KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV
knn_cv.fit(x,y)

#%% print hyperparameter KNN algoritmasındaki K degeri
print("tuned hyperparameter K: ",knn_cv.best_params_) #ayarlanmış değerde en iyi parametreyi seçer
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_) # en iyi accuracy değerini verir

# %% Grid search CV with logistic regression

x = x[:100,:] # ilk 100 değeri al
y = y[:100] 

from sklearn.linear_model import LogisticRegression
# buradaki c büyük seçilirse aşırı ezberleme olur eğer küçük seçilirse modeli hiç ezberleyememe olur
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

# en yüksek accuracy hesapladı
print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)
