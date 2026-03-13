import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

plt.style.use("classic")

df = pd.read_csv(r"C:\Users\AK17\Downloads\USA_Housing.csv")

print(df.head())


#reprocessing
#                   handling out layer
mask=(df["Avg. Area Income"]>0)&(df["Area Population"]>0)

f = df.drop(columns=["Price","Address"])[mask]
t = df["Price"][mask]


#                    handling missvalue
f=f.fillna(f.mean())
t=t.fillna(t.mean())


#                   feature engineering  (حل مشكلة underfitting)

f["Income_per_Pop"]=f["Avg. Area Income"]/f["Area Population"]
f["Rooms_per_Bed"]=f["Avg. Area Number of Rooms"]/f["Avg. Area Number of Bedrooms"]
f["Age2"]=f["Avg. Area House Age"]**2

# feature engineering اضافية لتقليل error
f["Income_per_Room"]=f["Avg. Area Income"]/f["Avg. Area Number of Rooms"]
f["Population_per_Room"]=f["Area Population"]/f["Avg. Area Number of Rooms"]
f["Bedrooms_per_Pop"]=f["Avg. Area Number of Bedrooms"]/f["Area Population"]
f["Rooms2"]=f["Avg. Area Number of Rooms"]**2
f["Income2"]=f["Avg. Area Income"]**2
f["Population2"]=f["Area Population"]**2
f["Income_Pop"]=f["Avg. Area Income"]*f["Area Population"]


#                    normalization <zscore norm>

mean_vals = f.mean()
std_vals = f.std()

f = (f - mean_vals) / std_vals


#plan model evaluations tratege
#                     train test split

num_7 = int(f.shape[0] * 0.7)

f_train=f.iloc[:num_7,:]
t_train=t.iloc[:num_7]

f_test=f.iloc[num_7:,:]
t_test=t.iloc[num_7:]


# kfold cross vald
k=KFold(n_splits=10,shuffle=True,random_state=42)


loops=[]
errors=[]
error_model=[]
paramter=[]

eps = 1e-4
lam = 0.1


#split data to learning and test

for tr_indx, te_index in k.split(f_train):

    f_tr = f_train.iloc[tr_indx].values
    f_te = f_train.iloc[te_index].values

    t_tr = t_train.iloc[tr_indx].values
    t_te = t_train.iloc[te_index].values

    cost2=0
    c=0

    alpha=0.003
    iteration=6000

    w=np.zeros(f_tr.shape[1])
    b=0
    
    error=[]
    loop=[]

    #training
    for i in range(iteration):

        y_pred=np.dot(f_tr,w)+b

        #costfun
        cost=(1/(2*f_tr.shape[0]))*(np.sum((t_tr-y_pred)**2)) + (lam/(2*f_tr.shape[0]))*np.sum(w**2)

        #gradiant decent 
        dendriative_w = (1/f_tr.shape[0]) * np.dot(f_tr.T, (y_pred - t_tr)) + (lam/f_tr.shape[0])*w
        dendriative_b = (1/f_tr.shape[0]) * np.sum(y_pred - t_tr)

        w=w-alpha*dendriative_w
        b=b-alpha*dendriative_b

        #linear cerve
        if i%10==0:
            error.append(cost)
            loop.append(i)

        #early stop
        if i>10:
            if abs(cost-cost2)<eps:
                c+=1
            else:
                c=0

        if c==10:
            break

        cost2=cost
       
    pred=np.dot(f_te,w)+b

    print(f"the cost of fold : {cost2}")

    error_model.append(np.sqrt(mean_squared_error(t_te,pred)))

    paramter.append([w,b])

    errors.append(error)
    loops.append(loop)


min_error_index=error_model.index(min(error_model))

w=paramter[min_error_index][0]
b=paramter[min_error_index][1]


#evaluation

model_pred=np.dot(f_test.values,w)+b

print("the error of fold test :", error_model)
print(f"index is : {min_error_index}")

print(f"mean error test : {np.sqrt(mean_squared_error(t_test,model_pred))}")


#draw

fig, ax = plt.subplots(2,5,figsize=(18,8))

for i in range(10):
    r=i//5
    c=i%5
    ax[r,c].set_title(f"liner cerve fold {i+1}")
    ax[r,c].plot(loops[i],errors[i],"--")

plt.savefig("cost for fold")
plt.figure()

plt.scatter(t_test, model_pred)
plt.plot([t_test.min(), t_test.max()],
         [t_test.min(), t_test.max()],
         'r--')

plt.xlabel("Actual")
plt.ylabel("Predicted")

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump((w, b, mean_vals, std_vals), file)

plt.show()