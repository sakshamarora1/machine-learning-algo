import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
pd.set_option('display.max_columns',None)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999,inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['Label'] = df[forecast_col].shift(-forecast_out)
print(forecast_out)

X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['Label'])
print(len(X),len(y))

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs= -1)
clf.fit(X_train,y_train)

# used pickling so as to not train the model every time the program is executed

# with open('linreg.pickle','wb') as f:
#     pickle.dump(clf,f)

# pickle_in = open('linreg.pickle','rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)

print('Accuracy: ',accuracy*100,'%')

forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# This will show the prediction graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()