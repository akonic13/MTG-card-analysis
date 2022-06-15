import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

'''
Clean and organize data
'''
df = pd.read_csv('card_data_test.csv')
df =df.replace({'Color': 'UBR,'},'UBR')
df = df.replace({'Color':'BU'},'UB')
df = df.replace({'Color':'GB'},'BG')
df =df.replace({'Color': 'UBR,'},'UBR')
df = df.replace({'Color':'BU'},'UB')
df = df.replace({'Color':'GB'},'BG')

#drop cards over $30
df2 = df[df['Price']<=30.00]
df2 = df2[df2['Rarity']!='S']

#df for rarities
dfC = df2[df2['Rarity']=='C']
dfU = df2[df2['Rarity']=='U']
dfR = df2[df2['Rarity']=='R']
dfM = df2[df2['Rarity']=='M']

'''
Count occurences of the 7 card types
'''
def typer(cardtype):
    if cardtype.find(_) != -1:
        return 1
    else:
        return 0

types = ['Land','Creature','Artifact','Enchantment','Instant','Sorcery','Planeswalker']
typecount = []
for i,_ in enumerate(types):
    typecount.append(float(df2['Card_type1'].apply(typer).sum()))
    df2[_] = df2['Card_type1'].apply(typer)
    
dftypes = pd.DataFrame(data={'Type':types,'Count':np.array(typecount)})
dftypes = dftypes.sort_values(by='Count')

'''
Transition color identities to numeric values
'''
def colorid(cardcolor):
    white = []
    blue = []
    black = []
    red = []
    green = []
    colorless = []
    for _ in cardcolor:
        if _.find('W') != -1:
            W = 1
        elif _.find('W') == -1:
            W = 0
        if _.find('U') != -1:
            U = 1
        elif _.find('U') == -1:
            U = 0
        if _.find('B') != -1:
            B = 1
        elif _.find('B') == -1:
            B = 0
        if _.find('R') != -1:
            R = 1
        elif _.find('R') == -1:
            R = 0
        if _.find('G') != -1:
            G = 1
        elif _.find('G') == -1:
            G = 0
        if _.find('colorless') != -1:
            C = 1
        elif _.find('colorless') == -1:
            C = 0
            
        white.append(W)
        blue.append(U)
        black.append(B)
        red.append(R)
        green.append(G)
        colorless.append(C)
            
    return white,blue,black,red,green,colorless

W,U,B,R,G,C = colorid(df2['Color'].tolist())
df2['W'] = np.array(W)
df2['U'] = np.array(U)
df2['B'] = np.array(B)
df2['R'] = np.array(R)
df2['G'] = np.array(G)
df2['C'] = np.array(C)

'''
Transfer rarities to numerics
'''
def rarity(rarity):
    if rarity == 'C':
        return 1
    if rarity == 'U':
        return 2
    if rarity == 'R':
        return 3
    if rarity == 'M':
        return 4

df2['Rarity_num'] = df2['Rarity'].apply(rarity)

'''
Plots of analyses
'''
type frequencies
plt.figure(figsize=(15,8),dpi=200)
sns.barplot(x='Type',y='Count',data=dftypes)
plt.xticks(rotation=(25))
plt.title('Frequency of the seven card supertypes')
plt.savefig('Supertype_frequency.png')

#price distributions
fig,axes = plt.subplots(2,2,figsize=(12,8),dpi=200)
sns.kdeplot(ax=axes[0,0],x=df2[df2['Rarity']=='C']['Price'],fill=True,color='black')
axes[0,0].set_xlim([0,3])
axes[0,0].set_title('Commons')

sns.kdeplot(ax=axes[0,1],x=df2[df2['Rarity']=='U']['Price'],fill=True,color='slategrey')
axes[0,1].set_xlim([0,5])
axes[0,1].set_title('Uncommons')

sns.kdeplot(ax=axes[1,0],x=df2[df2['Rarity']=='R']['Price'],fill=True,color='darkgoldenrod')
axes[1,0].set_xlim([0,20])
axes[1,0].xaxis.set_ticks(np.arange(0,21,5))
axes[1,0].set_title('Rares')

sns.kdeplot(ax=axes[1,1],x=df2[df2['Rarity']=='M']['Price'],fill=True,color='crimson')
axes[1,1].set_xlim([0,30])
axes[1,1].xaxis.set_ticks(np.arange(0,31,5))
axes[1,1].set_title('Mythics')

fig.suptitle('Price distributions')
fig.tight_layout()
plt.savefig('prices.png')

plt.figure(figsize=(15,8),dpi=200)
sns.kdeplot(data=df2,x='Price',fill=True)
plt.xlim([0,30])
plt.title('Price Distribution of all cards under $30')
plt.savefig('price_dist_all.png')

color combo frequencies
plt.figure(figsize=(12,8),dpi=200)
sns.countplot(x=sorted(df2['Color'],key=len),data=df2)
plt.xticks(rotation=(85))
plt.title('Frequency of the 32 color identities in MTG')
plt.savefig('ident_freq.png')

#by rarity
fig,axes = plt.subplots(2,2,figsize=(12,8),dpi=200)
sns.countplot(ax=axes[0,0],x=sorted(dfC['Color'],key=len),data=df2)
axes[0,0].tick_params(axis='x',rotation=(85))
axes[0,1].set_title('Uncommons')
axes[0,0].set_title('Commons')

sns.countplot(ax=axes[0,1],x=sorted(dfU['Color'],key=len),data=df2)
axes[0,1].tick_params(axis='x',rotation=(85))
axes[1,0].set_title('Rares')

sns.countplot(ax=axes[1,0],x=sorted(dfR['Color'],key=len),data=df2)
axes[1,0].tick_params(axis='x',rotation=(85))
axes[1,1].set_title('Mythics')

sns.countplot(ax=axes[1,1],x=sorted(dfM['Color'],key=len),data=df2)
axes[1,1].tick_params(axis='x',rotation=(85))

fig.suptitle('Frequency of color identities by rarity')
fig.tight_layout()
plt.savefig('ident_freq_rarity.png')

#correlation strengths
plt.figure(figsize=(12,8),dpi=200)
sns.barplot(x=df2.corr()['Price'].sort_values().index[:-1],y=df2.corr()['Price'].sort_values()[:-1])
plt.xticks(rotation=(65))
plt.title('Price vs Correlation')
plt.yticks(np.arange(-.10,.45,.05))
plt.savefig('Price_corrs.png')

'''
Machine learning: 
    Initialize X and y data and perform train test split and scaling
'''
df3 = df2.drop(['Card_Name','Set','Card_type1','Card_type2','Mana_Cost','Color','Rarity'],axis=1)

X = df3.drop('Price',axis=1)
#X['Modal'] = pd.get_dummies(X['Modal'],drop_first=True)
X['Modal'] = pd.get_dummies(X['Modal'])
X['Modal'] = 0
#X['Phyrexian_Mana'] = pd.get_dummies(X['Phyrexian_Mana'],drop_first=True)
X['Phyrexian_Mana'] = pd.get_dummies(X['Phyrexian_Mana'])
X['Phyrexian_Mana'] = 0
y = df3['Price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .15,random_state=42)

scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

'''
First Model: 
    Linear Regression (RidgeCV)
'''
ridge_model = RidgeCV(alphas = np.arange(0,201,1),cv=15,scoring='neg_mean_squared_error')
ridge_model.fit(X_train_scale,y_train)
print('Ridge Done')
price_pred_ridge = ridge_model.predict(X_test_scale)

MAE_ridge = mean_absolute_error(y_test,price_pred_ridge)
RMSE_ridge = mean_squared_error(y_test, price_pred_ridge,squared=False)

'''
Second Model:
    Support Vector Regression
'''
svr = SVR()

params ={'C':np.arange(100,111),'gamma':['auto','scale']}

gridSVR = GridSearchCV(svr, param_grid=params,cv=15,verbose=0)
gridSVR.fit(X_train_scale,y_train)
print('SVR done')
price_pred_svr = gridSVR.predict(X_test_scale)

MAE_svr = mean_absolute_error(y_test, price_pred_svr)
RMSE_svr = mean_squared_error(y_test, price_pred_svr,squared=False)

'''
Third Model:
    Random Forest Regression
'''
rfr = RandomForestRegressor()
params3 = {'n_estimators': list(np.arange(1,101)),'max_depth':list(np.arange(1,5))}

gridrfr = GridSearchCV(rfr, param_grid=params3,cv=10,verbose=0)
gridrfr.fit(X_train_scale,y_train)

price_pred_rfr = gridrfr.predict(X_test_scale)

MAE_rfr = mean_absolute_error(y_test, price_pred_rfr)
RMSE_rfr = mean_squared_error(y_test, price_pred_rfr)

'''
Third Model:
    Neural Network regression
'''
scaler2 = MinMaxScaler()
X_train_scale = scaler2.fit_transform(X_train)
X_test_scale = scaler2.transform(X_test)

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(5,activation='relu'),
    tf.keras.layers.Dense(5,activation='relu'),
    tf.keras.layers.Dense(5,activation='relu'),
    tf.keras.layers.Dense(1,activation='linear')
    ])
model.compile(loss='mse',optimizer=tf.optimizers.Adam(learning_rate=.01),metrics=['mse'])

model.fit(X_train_scale,y_train,epochs=200,verbose=0)
model.save('NeuralNetworkModel')
price_pred_nnr = tf.squeeze(tf.constant(model.predict(X_test_scale)))
#
MAE = mean_absolute_error(y_test, price_pred_nnr)
RMSE = np.sqrt(tf.metrics.mean_squared_error(y_test,price_pred_nnr))
HE = tf.keras.losses.Huber(delta=1)
HE = HE(y_test,y_pred).numpy()

plt.figure()
pd.DataFrame(fits.history).plot()
model.summary()
plot_model(model=model,show_shapes=True,dpi=200)
