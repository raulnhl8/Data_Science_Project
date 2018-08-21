import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Pre-processing
# Retira colunas que possuem pelo menos 33% dos valores NaN
df_train.dropna(axis=1, thresh=1000, inplace=True)
df_test.dropna(axis=1, thresh=1000, inplace=True)
# Retira coluna com id
df_train.drop(df_train.columns[[0]], axis=1, inplace=True)
df_test.drop(df_test.columns[[0]], axis=1, inplace=True)
# Preenche o resto dos valores NaN com o valor mais frequente naquela coluna
df_train = df_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
df_test = df_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Visualizacao dos dados e analise
#corrmat = df_train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True)

# Retira variaveis redundates baseado na analise feita
drop_variables = ['1stFlrSF', 'TotRmsAbvGrd', 'GarageArea' , 'GarageYrBlt']
df_train.drop(drop_variables, axis=1, inplace=True)
df_test.drop(drop_variables, axis=1, inplace=True)

# Removendo pontos reconhecidos como outliers
df_train.drop(df_train.index[523], inplace=True)
df_train.drop(df_train.index[1297], inplace=True)

# Plot das variaveis com maior correlacao, consideradas mais relevantes
#relevant_var = ['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageCars']
#for var in relevant_var:
#    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Divide dados entre variaveis dependentes e independentes
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, 70:]
X_test = df_test

# Aplicar log na variavel para tentar aproximar ela da curva normal, como descrito no relatorio
y_train = np.log1p(y_train)

# Codificar os labels e separar em colunas diferentes
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Como algumas categorias estavam presentes no X_train e nao estavam no X_test,
#   algumas colunas nao foram criadas no X_test, entao estas colunas foram preenchidas com 0
train_to_test = list(set(X_train.columns) - set(X_test.columns))
df_ = pd.DataFrame(data= np.zeros((X_test.shape[0],len(train_to_test))), columns=train_to_test)
X_test = pd.concat([X_test, df_], axis=1)

# Ordena as colunas de ambos dataFrames, para garantir que estejam na mesma posicao
X_train.sort_index(axis=1, inplace=True)
X_test.sort_index(axis=1, inplace=True)

# Normaliza o valor das variaveis
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Preparing models
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#function
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

avModels = AveragingModels(models = (ENet, GBoost, KRR, lasso))
avModels.fit(X_train, y_train.values)
stacked_train_pred = avModels.predict(X_train)
stacked_pred = np.expm1(avModels.predict(X_test))

model_xgb.fit(X_train, y_train)
xgb_train_pred = model_xgb.predict(X_train)
xgb_pred = np.expm1(model_xgb.predict(X_test))

y_pred = stacked_pred*0.6 + xgb_pred*0.4


#print(type(y_pred))

# Armzena resultados em arquivos
file = open("pred.csv", "w")
idCont = 1461
file.write("Id,SalePrice\n")
for pred in y_pred:
    file.write(str(idCont) +","+ str(pred)+"\n")
    idCont+=1
print("close")
file.close()
