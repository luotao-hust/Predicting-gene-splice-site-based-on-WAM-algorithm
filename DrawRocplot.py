# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import auc

data=pd.read_csv("fpr_tpr_threshold.csv",header=None)
#data=pd.read_csv("ROC_frame_Beyess_test.csv",header=None)
print("data")
print(data)
#data.sort(columns = [0],axis = 0,ascending = True)  ##一定要记得排序，否则无法画出一一对应的图

fpr=data[0].to_list()
tpr=data[1].to_list()
C=data[2].to_list()

accuracy_index = []
for i in range(len(fpr)):
    accuracy_index.append(tpr[i]+(1-fpr[i])-1)
accu_max = max(accuracy_index)
accu_index = accuracy_index.index(accu_max)
fpr_m = fpr[accu_index]
tpr_m = tpr[accu_index]
C_value = C[accu_index]


#fpr.sort()    #一定要记得排序，否则无法画出一一对应的图
#tpr.sort()
'''
auc = 0
for i in range(len(data.index)-1):
   h = fpr[i+1]-fpr[i]  #矩形的高
   l1 = tpr[i]
   l2 = tpr[i+1]
   auc += (l1+l2)*h/2
'''
roc_auc = auc(fpr, tpr)   #计算auc
plt.plot(fpr,tpr,linewidth=2,label="ROC",)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("false presitive rate")
plt.ylabel("true presitive rate")
plt.ylim(0,1)
plt.xlim(0,1)
plt.legend(loc=4)#图例的位置plt.show()
plt.annotate("accuracy index:{0}\n fpr:{1}\n tpr{2}\n auc:{3}\n C:{4}".format(accu_max,fpr_m,tpr_m,roc_auc,C_value) , xy=(fpr_m,tpr_m))
plt.annotate("accuracy index:{0}\n fpr:{1}\n tpr{2}\n auc:{3}\n ".format(accu_max,fpr_m,tpr_m,roc_auc,) , xy=(fpr_m,tpr_m))

plt.show()

