
# coding: utf-8

# In[1]:

from initNN import *
from sklearn.metrics import mean_squared_error
def mean_percentage_error(y_pred,y_actual):
    'Calculate the mean percentage absolute error',
    n = y_pred.shape[0]
    temp = [np.abs((i-j)/j) for i,j in zip(y_pred,y_actual)],
    return (1.0/n) * np.sum(temp)
def getPredict(start_idx,end_idx,type_mode,n_input=15):
    ax = pl.subplot()
    nn_mode = LoadParam(type_mode,n_input)
    X_test,y_test = nn_mode.generate((start_idx,end_idx))
    ax.set_color_cycle(['blue','red'])
    ax.plot(nn_mode.predict(X_test),'--',label='Predict')
    ax.plot(y_test,label='Actual')
    pl.show()


# In[8]:

list_nresult = []
for n_input in np.arange(2,21):
#     n_input=15
    temp = []
    nn = LoadParam("GN",n_input)
    gn = LoadParam("GN",n_input,1)
    i=46
    skip_list = 3
    X_test,y_test = nn.generate((i,i+skip_list))
    X_ptest, y_ptest = gn.generate((i,i+skip_list))
    gn_pred = gn.convert(gn.predict(X_ptest))
    nn_pred = nn.convert(nn.predict(X_test))

    temp.append(np.sqrt(mean_squared_error(nn_pred,y_test)))
    temp.append(mean_absolute_error(nn_pred,y_test))
    temp.append(mean_percentage_error(y_test,nn_pred))

    temp.append(np.sqrt(mean_squared_error(gn_pred,y_ptest)))
    temp.append(mean_absolute_error(gn_pred,y_ptest))
    temp.append(mean_percentage_error(y_ptest,gn_pred))

    list_nresult.append(temp)


# In[7]:

# ax = pl.subplot()
# ax.set_color_cycle(['blue','red'])
# ax.plot(nn_pred,'--',label='Predict')
# ax.plot(y_test,label='Actual')
# pl.show()


# In[15]:

# results = np.load("experiement_12-1-mae_window.npz")
# results.files


# In[9]:

# ax = pl.subplot()
# ax.plot(gn_pred)
# ax.plot(y_test)
# pl.show()
print pd.DataFrame(list_nresult)


# In[ ]:



