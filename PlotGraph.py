
# coding: utf-8

# In[2]:

from TrafficFeeder import *
from sklearn.metrics import mean_squared_error
def mean_percentage_error(y_pred,y_actual):
    'Calculate the mean percentage absolute error',
    n = y_pred.shape[0]
    temp = [np.abs((i-j)/j) for i,j in zip(y_pred,y_actual)],
    return (1.0/n) * np.sum(temp)


# In[3]:

list_nresult = []
for n_input in np.arange(2,21):
    temp = []
    nn = LoadParam("NN",n_input)
    gn = LoadParam("GN",n_input)
    i=46
    skip_list = 3
    X_test,y_test = nn.generate((i,i+skip_list))
    X_ptest, y_ptest = gn.generate((i,i+skip_list))
    gn_pred = gn.convert(gn.predict(X_test))
    nn_pred = nn.convert(nn.predict(X_ptest))
    
    temp.append(np.sqrt(mean_squared_error(nn_pred,y_test)))
    temp.append(mean_absolute_error(nn_pred,y_test))
    temp.append(mean_percentage_error(y_test,nn_pred))
 
    temp.append(np.sqrt(mean_squared_error(gn_pred,y_ptest)))
    temp.append(mean_absolute_error(gn_pred,y_ptest))
    temp.append(mean_percentage_error(y_ptest,gn_pred))
    
    list_nresult.append(temp)


# In[15]:

results = np.load("experiement_12-1-mae_window.npz")
results.files


# In[4]:

# ax = pl.subplot()
# ax.plot(gn_pred)
# ax.plot(y_test)
# pl.show()
pd.DataFrame(list_nresult)


# In[ ]:



