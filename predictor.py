import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#import sales data
df=pd.read_excel("ciro_analizi_.xlsm",'test')


X= df.iloc[:,1:6]
y=df['Ciro'].copy()
    
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)


y_test2 = y_test.as_matrix()

# initiate the model
regr = linear_model.LinearRegression()

#print(dir(regr))
# fit and predict the model
regr.fit(X_train, y_train)
results= regr.predict(X_test)
fark=(results-y_test2)/y_test2


# plot the deviation between prediction and test data
print (fark)
plt.figure()
plt.plot(fark,'x',color='red')
plt.show()
