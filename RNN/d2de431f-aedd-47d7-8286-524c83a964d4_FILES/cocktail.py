# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:49:40 2020

@author: YU TANG
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from sklearn.preprocessing import *
from keras.preprocessing import *

data_1=pd.read_csv('cocktails.csv')
#X_train=data_1.drop(['Bartender','Bar/Company','Location','Garnish','Glassware','Preparation','Notes'],axis=1).values
#Y_train=data_1['Cocktail Name'].values

X_train=data_1['Ingredients'].values
#Y_train=[1:655]
#print(X_train)

X_test=data_1['Ingredients'][0]
#print(X_test)

Y=pd.get_dummies(data_1['Cocktail Name'])
print(Y.head(2))


for i in range (687):
    Y1_train[i]=i
    

from keras.preprocessing.text import Tokenizer

token = Tokenizer(num_words=20000)
token.fit_on_texts(X_train)
#token.fit_on_texts(Y_train)

#print(token.word_index)
#print(X_train)
X1_train = token.texts_to_sequences(X_train)
#Y1_train = token.texts_to_sequences(Y_train)

X1_test = token.texts_to_sequences(X_test)

max_seq_len = max([
    len(seq) for seq in X1_train])
print(max_seq_len)

max_seq_len0 = max([
    len(seq) for seq in Y1_train])

X1_train = sequence.pad_sequences(X1_train, max_seq_len)
#Y1_train = sequence.pad_sequences(Y1_train, max_seq_len0)

print(Y1_train.shape)

X1_test = sequence.pad_sequences(X1_test, max_seq_len)

from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(input_dim=20000, 
                    output_dim=128,
                    input_length=43))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=256,
                activation='relu' ))
model.add(Dropout(0.2))
model.add(Dense(units=684,
                activation='sigmoid' ))
model.summary()

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='nadam')
#進行訓練
#batch_size：每一批次訓練100筆資料
#epochs：執行10個訓練週期
#verbose：顯示每次的訓練過程
#validation_split：測試資料的比例
his =model.fit(X1_train, Y,batch_size=100,
                         epochs=10,verbose=2,validation_split=0.25)

import matplotlib.pyplot as plt

plt.plot(his.history['loss'])       
plt.plot(his.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Validation Mae')

#plt.savefig(fn)  

plt.show()
  
pred =model.predict(X1_test)
print(pred)

"""                    
#評估訓練模型的準確率
acu = model.evaluate(x_test, y_test, verbose=1)
acu[1]
"""
"""
index_to_label = {v: k for k, v in Y.items()}

test['Category'] = [index_to_label[idx] for idx in np.argmax(pred, axis=1)]

submission = test.loc[:, ['Category']].reset_index()

submission.columns = ['Id', 'Category']
"""