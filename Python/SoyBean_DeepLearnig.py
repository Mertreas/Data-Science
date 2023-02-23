#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Formação Cientista de Dados - Fernando Amaral e Jones Granatyr
# Faça Você Mesmo ML e RNA


# In[4]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[6]:


base = pd.read_csv(r'C:\Users\leo_b\Documents\Cientista de Dados\34.Prática em Python\dados\soybean.csv')
base


# In[7]:


#Criando a variável X que terá os atributos previsores
X = base.iloc[:, 0:35].values
#criando a variável y que terá os atributos previsiveis
y = base.iloc[:, 35]
X


# In[8]:


#Label Encoder irá atribuir números para distinguir caracteristicas da coluna
labelencoder = LabelEncoder()

for x in range(35):
    X[:, x] = labelencoder.fit_transform(X[:, x])


# In[15]:


# Divisão da base em treino e teste (70% para treinamento e 30% para teste)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
floresta = RandomForestClassifier(n_estimators = 100)
floresta.fit(X_treinamento, y_treinamento)


# In[18]:


#previsões
previsoes = floresta.predict(X_teste)
previsoes


# In[19]:


#Matrix de confunsão para saber o desempenho
matriz = confusion_matrix(y_teste, previsoes)
matriz


# In[22]:


#Score de acertos
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_acerto


# In[23]:


#Score de erros
taxa_erro = 1 - taxa_acerto
taxa_erro

