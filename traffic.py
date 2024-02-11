import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import math
from subprocess import check_output
import statistics
df = pd.read_csv('cs448b_ipasn.csv')
df['date']= pd.to_datetime(df['date'])
df = df.groupby(['date','l_ipn'],as_index=False).sum()
df['yday'] = df['date'].dt.dayofyear
df['wday'] = df['date'].dt.dayofweek


ip0 = df[df['l_ipn']==0]
max0 = np.max(ip0['f'])
ip1 = df[df['l_ipn']==1]
max1 = np.max(ip1['f'])
ip2 = df[df['l_ipn']==2]
max2 = np.max(ip2['f'])
ip3 = df[df['l_ipn']==3]
max3 = np.max(ip3['f'])
ip4 = df[df['l_ipn']==4]
max4 = np.max(ip4['f'])
ip5 = df[df['l_ipn']==5]
max5 = np.max(ip5['f'])
ip6 = df[df['l_ipn']==6]
max6 = np.max(ip6['f'])
ip7 = df[df['l_ipn']==7]
max7 = np.max(ip7['f'])
ip8 = df[df['l_ipn']==8]
max8 = np.max(ip8['f'])
ip9 = df[df['l_ipn']==9]
max9 = np.max(ip9['f'])


mean_ip0 = statistics.mean(ip0['f'])
var_ip0 = statistics.variance(ip0['f']) 
prob_0 =[]
for i in ip0['f']:
 
    a = (math.exp(((i)-mean_ip0)**2/(-2*var_ip0)))/(math.sqrt(2*math.pi*var_ip0))
    prob_0.append([i,a])
prob_0.sort(key = lambda x: x[1])
print(prob_0)
anomalies_0 = [prob_0[0][1]]
j_0=0
while prob_0[j_0][0] < 2 * prob_0[(j_0)+1][0]:
    anomalies_0.append(prob_0[(j_0)+1][1])
    (j_0)+=1    
#get the days as required
plt.plot(ip0['yday'],ip0['f'])
plt.show()

mean_ip1 = statistics.mean(ip0['f'])
var_ip1 = statistics.variance(ip0['f']) 
prob_1 =[]
for i in ip1['f']:
 
    a = (math.exp(((i)-mean_ip1)**2/(-2*var_ip1)))/(math.sqrt(2*math.pi*var_ip1))
    prob_1.append([i,a])
prob_1.sort(key = lambda x: x[1])

anomalies_1 = [prob_1[0][1]]
j_1=0
while prob_1[j_1][0] < 2 * prob_1[(j_1)+1][0]:
    anomalies_1.append(prob_1[(j_1)+1][1])
    (j_1)+=1    
#get the days as required
plt.plot(ip1['yday'],ip1['f'])
plt.show()

mean_ip2 = statistics.mean(ip0['f'])
var_ip2 = statistics.variance(ip0['f']) 
prob_2 =[]
for i in ip2['f']:
 
    a = (math.exp(((i)-mean_ip2)**2/(-2*var_ip2)))/(math.sqrt(2*math.pi*var_ip2))
    prob_2.append([i,a])
prob_2.sort(key = lambda x: x[1])

anomalies_2 = [prob_2[0][1]]
j_2=0
while prob_2[j_2][0] < 2 * prob_2[(j_2)+1][0]:
    anomalies_2.append(prob_2[(j_2)+1][1])
    (j_2)+=1    
#get the days as required
plt.plot(ip2['yday'],ip2['f'])
plt.show()


mean_ip3 = statistics.mean(ip0['f'])
var_ip3 = statistics.variance(ip0['f']) 
prob_3 =[]
for i in ip3['f']:
 
    a = (math.exp(((i)-mean_ip3)**2/(-2*var_ip3)))/(math.sqrt(2*math.pi*var_ip3))
    prob_3.append([i,a])
prob_0.sort(key = lambda x: x[1])

anomalies_3 = [prob_3[0][1]]
j_3=0
while prob_3[j_3][0] < 2 * prob_3[(j_3)+1][0]:
    anomalies_3.append(prob_3[(j_3)+1][1])
    (j_3)+=1    
#get the days as required
plt.plot(ip3['yday'],ip3['f'])
plt.show()


mean_ip4 = statistics.mean(ip0['f'])
var_ip4 = statistics.variance(ip0['f']) 
prob_4 =[]
for i in ip4['f']:
 
    a = (math.exp(((i)-mean_ip4)**2/(-2*var_ip4)))/(math.sqrt(2*math.pi*var_ip4))
    prob_4.append([i,a])
prob_4.sort(key = lambda x: x[1])

anomalies_4 = [prob_4[0][1]]
j_4=0
while prob_4[j_4][0] < 2 * prob_4[(j_4)+1][0]:
    anomalies_4.append(prob_4[(j_4)+1][1])
    (j_4)+=1    
#get the days as required
plt.plot(ip4['yday'],ip4['f'])
plt.show()


mean_ip5 = statistics.mean(ip0['f'])
var_ip5 = statistics.variance(ip0['f']) 
prob_5 =[]
for i in ip5['f']:
 
    a = (math.exp(((i)-mean_ip5)**2/(-2*var_ip5)))/(math.sqrt(2*math.pi*var_ip5))
    prob_5.append([i,a])
prob_5.sort(key = lambda x: x[1])

anomalies_5 = [prob_5[0][1]]
j_5=0
while prob_5[j_5][0] < 2 * prob_5[(j_5)+1][0]:
    anomalies_0.append(prob_5[(j_5)+1][1])
    (j_5)+=1    
#get the days as required
plt.plot(ip0['yday'],ip0['f'])
plt.show()


mean_ip6 = statistics.mean(ip0['f'])
var_ip6 = statistics.variance(ip0['f']) 
prob_6 =[]
for i in ip6['f']:
 
    a = (math.exp(((i)-mean_ip6)**2/(-2*var_ip6)))/(math.sqrt(2*math.pi*var_ip6))
    prob_6.append([i,a])
prob_6.sort(key = lambda x: x[1])

anomalies_6 = [prob_6[0][1]]
j_6=0
while prob_6[j_6][0] < 2 * prob_0[(j_6)+1][0]:
    anomalies_0.append(prob_6[(j_6)+1][1])
    (j_6)+=1    
#get the days as required
plt.plot(ip6['yday'],ip6['f'])
plt.show()


mean_ip7 = statistics.mean(ip0['f'])
var_ip7 = statistics.variance(ip0['f']) 
prob_7 =[]
for i in ip7['f']:
 
    a = (math.exp(((i)-mean_ip7)**2/(-2*var_ip7)))/(math.sqrt(2*math.pi*var_ip7))
    prob_7.append([i,a])
prob_7.sort(key = lambda x: x[1])
print(prob_7)
anomalies_7 = [prob_7[0][1]]
j_7=0
while prob_7[j_7][0] < 2 * prob_7[(j_7)+1][0]:
    anomalies_7.append(prob_7[(j_7)+1][1])
    (j_7)+=1    
#get the days as required
plt.plot(ip7['yday'],ip7['f'])
plt.show()


mean_ip8 = statistics.mean(ip0['f'])
var_ip8 = statistics.variance(ip0['f']) 
prob_8 =[]
for i in ip8['f']:
 
    a = (math.exp(((i)-mean_ip8)**2/(-2*var_ip8)))/(math.sqrt(2*math.pi*var_ip8))
    prob_0.append([i,a])
prob_8.sort(key = lambda x: x[1])
print(prob_0)
anomalies_8 = [prob_8[0][1]]
j_8=0
while prob_0[j_8][0] < 2 * prob_0[(j_8)+1][0]:
    anomalies_8.append(prob_8[(j_8)+1][1])
    (j_8)+=1    
#get the days as required
plt.plot(ip8['yday'],ip8['f'])
plt.show()


mean_ip9 = statistics.mean(ip9['f'])
var_ip9 = statistics.variance(ip9['f']) 
prob_9 =[]
for i in ip0['f']:
 
    a = (math.exp(((i)-mean_ip9)**2/(-2*var_ip9)))/(math.sqrt(2*math.pi*var_ip9))
    prob_9.append([i,a])
prob_9.sort(key = lambda x: x[1])

anomalies_9 = [prob_9[0][1]]
j_9=0
while prob_9[j_0][0] < 2 * prob_0[(j_9)+1][0]:
    anomalies_9.append(prob_9[(j_9)+1][1])
    (j_9)+=1    
#get the days as required
plt.plot(ip9['yday'],ip9['f'])
plt.show()
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#normalizing data for use in RNN




fv =[float(v)/float(max0) for v in ip0['f'].values]
ip0.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max1) for v in ip1['f'].values]
ip1.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max2) for v in ip2['f'].values]
ip2.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max3) for v in ip3['f'].values]
ip3.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max4) for v in ip4['f'].values]
ip4.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max5) for v in ip5['f'].values]
ip5.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max6) for v in ip6['f'].values]
ip6.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max7) for v in ip7['f'].values]
ip7.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max8) for v in ip8['f'].values]
ip8.loc[:,'f'] =np.array(fv).reshape(-1,1)
fv =[float(v)/float(max9) for v in ip9['f'].values]
ip9.loc[:,'f'] =np.array(fv).reshape(-1,1)

print(prob_0,prob_1,prob_2,prob_3,prob_4,prob_5,prob_6,prob_7,prob_8,prob_9)