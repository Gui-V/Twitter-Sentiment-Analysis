#%% Imports
!pip install textblob

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.figure_factory as ff
import seaborn as  sns
import pickle
import datetime as dt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
#from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import re
import emoji
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
#pip install emoji --upgrade
from nltk.corpus import stopwords
#pip install -U textblob
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import CountVectorizer

# %%
############### 2. DATA UNDERSTANDING #################
##### 2.1 Features base #####
corpus= pd.read_csv("Tweets.csv")
corpus.head()

#ver se há tweet_id repetidos e remove-los
corpus['tweet_id'] = corpus['tweet_id'].astype('category')
corpus.tweet_id.describe()
corpus.drop_duplicates(subset='tweet_id',inplace=True)

#ver os missing values
print(corpus.isnull().sum())

corpus.tweet_location.describe()
corpus.user_timezone.describe()

#retirar as features tweet_coord, negativereason_gold  e airline_sentiment_gold
#do dataset porque tem muitos missing values
((corpus.isnull()|corpus.isna()).sum()*100/corpus.index.size).round(2)
corpus=corpus.drop(columns=['tweet_coord', 'negativereason_gold', 'airline_sentiment_gold'])
corpus.head()


corpus.columns

# %%

#ANÁLISE DESCRITIVA
corpus.info()
summary=corpus.describe(include='all')
summary=summary.transpose()
summary.head(len(summary))

#O modelo não é balanceado
corpus.airline_sentiment.value_counts() / len(corpus)

#Frequência de observações com sentimento negativo, positivo e neutro
neg = len(corpus[corpus["airline_sentiment"] == "negative"])
pos = len(corpus[corpus["airline_sentiment"] == "positive"])
neu = len(corpus[corpus["airline_sentiment"] == "neutral"])
dist = [graph_objs.Bar(x=["negative","neutral","positive"],y=[neg, neu, pos],)]
plotly.offline.iplot({"data":dist, "layout":graph_objs.Layout(title="Sentiment Type in Dataset")})# %%
plt.show()

# % por sentimento
(corpus['airline_sentiment'].value_counts()*100/corpus.index.size).round(2)

# % por companhia - A United tem mais review e a Virgin é a que tem menos
(corpus['airline'].value_counts()*100/corpus.index.size).round(2)

#Pie chart of tweets frequency for each airline
lista=(corpus['airline'].value_counts()/corpus.index.size).round(4)
labels = ['United','US Airways','American','Southwest','Delta','Virgin America']
sizes = lista
fig1, ax1 = plt.subplots(figsize=(6.5, 6))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  
plt.show()


#Frequência das razões para comentários negativos
corpus.negativereason.value_counts().plot(kind='barh',figsize=(8,8))
plt.xlabel('Frequency')
plt.show()

#boxplot das features numericas
numerical=corpus.select_dtypes(include=[np.number]).columns.tolist()
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
for var, subplot in zip(corpus[numerical], ax.flatten()):
    sns.boxplot(corpus[var], ax=subplot)
plt.show()

#Há várias pessoas/agências a fazer mais do que um tweet
corpus['name'] = corpus['name'].astype('category')
corpus.name.describe()

#bins para retweet - A probabilidade de retweet é maior quando a apreciação é negativa
corpus['bins_retweet_count']=pd.cut(x=corpus['retweet_count'], bins=[-1,0,44], labels=['nao retweet','retweet'])
table1 = pd.crosstab(index=corpus['bins_retweet_count'], columns=corpus['airline_sentiment'], margins=True, margins_name='Total')
table1
table2=pd.crosstab(index=corpus['bins_retweet_count'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table2

corpus['bins_retweet']=pd.cut(x=corpus['retweet_count'], bins=[-1,0,1,2,44], labels=['0','1','2', '3+'])
table3 = pd.crosstab(index=corpus['bins_retweet'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table3
table3 = pd.crosstab(index=corpus['bins_retweet'], columns=corpus['airline_sentiment'], margins=True, margins_name='Total')
table3
table4 = pd.crosstab(index=corpus['bins_retweet'], columns=corpus['airline_sentiment'])
table4.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)
plt.show()



##### 2.2 Análise descritiva comparada das companhias aéreas #####

#crosstab por sentimento e companhia
table5 = pd.crosstab(index=corpus['airline'], columns=corpus['airline_sentiment'], margins=True, margins_name='Total')
table5

# Percentagem da frequência de tweets negativos, positivos e neutros por companhia aérea
mask_neg = corpus['airline'][corpus.airline_sentiment=='negative']
mask_neut = corpus['airline'][corpus.airline_sentiment=='neutral']
mask_pos = corpus['airline'][corpus.airline_sentiment=='positive']
airline_tweet_totals = corpus.airline.value_counts()
neg = mask_neg.value_counts(); neut = mask_neut.value_counts(); pos = mask_pos.value_counts()
neg.sort_index(inplace=True); pos.sort_index(inplace=True); 
neut.sort_index(inplace=True); airline_tweet_totals.sort_index(inplace=True);
perc_neg = (neg/airline_tweet_totals)*100
perc_pos = (pos/airline_tweet_totals)*100
perc_neut = (neut/airline_tweet_totals)*100
width=0.25
plt.figure(figsize=(5,4))
perc_neg.plot(kind='bar', color='grey', width=width, position=1)
perc_neut.plot(kind='bar', color='green', width= width, position=0.5)
perc_pos.plot(kind='bar', color='pink', width=width, position=0)
plt.legend(('negative', 'neutral', 'positive'))
plt.grid(True)
plt.xticks(rotation=90)
plt.ylabel('(%)')
plt.tight_layout()
plt.show()
#ou
table6= pd.crosstab(corpus.airline, corpus.airline_sentiment).apply(lambda x: x / x.sum() * 100, axis=1).plot(kind='bar',figsize=(8,8),stacked=True)
table6.set_ylabel('(%)')
table6.set_xlabel('')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()
#ou
table7 = pd.crosstab(index=corpus['airline'], columns=corpus['airline_sentiment'])
table7.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)
plt.show()


#Percentagem da frequência de cada label “negativereason” por companhia aérea
table8 = pd.crosstab(index=corpus['negativereason'], columns=corpus['airline']).apply(lambda r: r/r.sum(), axis=0)
table8



############### 3. DATA CLEANING #################
##### 2.1 Novas features #####


######## Nova Feature: len_text
#bins para len_text - textos mais compridos são mais provaveis em negativos, textos mais curtos são mais provaveis em neutros ou positivos
len_text=corpus.text.str.len()
corpus['len_text']=len_text
corpus['bins_len_text']=pd.cut(x=corpus['len_text'], bins=[0,77,114,136,200])
pd.crosstab(index=corpus['bins_len_text'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
#Falta fazer one hot encoding

#Distribuição de densidade do len_text por sentimento
sentiments = [ 'negative','neutral', 'positive']
for i in sentiments:
    subset = corpus[corpus['airline_sentiment'] == i]
    data=subset['len_text']
    data.plot(kind="kde")
plt.xlim(0, 175)
figsize=(5,6)
plt.legend(['negative','neutral', 'positive'])     
plt.show()


######### Nova Feature: num_oc_comp
#Funcoes para processar "@name"
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt   

#Substituir por@airlinename
regex = r"@(VirginAmerica|united|SouthwestAir|Delta|JetBlue|USAirways|AmericanAir)"
def text_replace(text):
    return re.sub(regex, '@airlinename', text, flags=re.IGNORECASE)
corpus['text_airlinename'] = corpus['text'].apply(text_replace)



#Nova feature - Nº Ocorrencias do nome da companhia aérea
corpus['num_oc_comp']=corpus.text_airlinename.apply(lambda x: x.count('@airlinename'))
corpus['num_oc_comp'].value_counts()
corpus['bins_num_oc_comp']=pd.cut(x=corpus['num_oc_comp'], bins=[-1,0,1,5], labels=['0','1','2+'])
pd.crosstab(index=corpus['bins_num_oc_comp'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)

######### Nova Feature: num_oc 
# Nº Ocorrencias do @handle

regex2 = r"@[\w]*"
def text_replace2(text):
    return re.sub(regex2, '@', text, flags=re.IGNORECASE)
#retirar @airlinename e expressao regular para @tag
corpus['text_s_airlinename']=np.vectorize(remove_pattern)(corpus['text_airlinename'], "@airlinename")
corpus['text_s_airlinename'] = corpus['text_s_airlinename'].apply(text_replace2)
#count @
corpus['num_oc']=corpus.text_s_airlinename.apply(lambda x: x.count('@'))
corpus['num_oc'].value_counts()
#análise gráfica
corpus['bins_num_oc']=pd.cut(x=corpus['num_oc'], bins=[-1,0,1,2,6], labels=['0','1','2','3+'])
pd.crosstab(index=corpus['bins_num_oc'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)

table9 = pd.crosstab(index=corpus['bins_num_oc'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=1)
table9
table9.plot(kind="bar", figsize=(8,8), stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

######### Nova Feature: num_Hash
# Captar Hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i,re.IGNORECASE)
        for idx,word in enumerate(ht):
            ht[idx]=word.lower()
            if word[-1]=='s':
                ht[idx]=word[:-1]
        lista=['UnitedAirline','united','Jetblue', 'AmericanAirline']
        #lista_lower=list(map(lambda x:x.lower(),lista))
        #ht_lower=list(map(lambda x:x.lower(),ht))
        #ht=[palavra for palavra in ht_lower if palavra not in lista_lower]
        lista_lower = {i.lower() for i in lista}
        for word in ht:
            if word.lower() in lista_lower:
                ht.remove(word)
        hashtags.append(ht)
    return hashtags

# Hashtags de negative tweets
HT_negative = hashtag_extract(corpus['text'][corpus['airline_sentiment'] == 'negative'])
# Hashtags de neutral tweets
HT_neutral = hashtag_extract(corpus['text'][corpus['airline_sentiment'] == 'neutral'])
# Hashtags de positive tweets
HT_positive = hashtag_extract(corpus['text'][corpus['airline_sentiment'] == 'positive'])

corpus.loc[corpus.loc[corpus.airline_sentiment=='negative'].index, 'Hash'] = HT_negative
corpus.loc[corpus.loc[corpus.airline_sentiment=='neutral'].index, 'Hash'] = HT_neutral
corpus.loc[corpus.loc[corpus.airline_sentiment=='positive'].index, 'Hash'] = HT_positive

#Nº de Hashtatgs
corpus['num_Hash']=corpus['Hash'].apply(lambda x: len(x) )

lista_hash_positivos=((corpus['Hash'][corpus['airline_sentiment'] == 'positive']).apply(', '.join)).value_counts()[1:]
lista_hash_negativos=((corpus['Hash'][corpus['airline_sentiment'] == 'negative']).apply(', '.join)).value_counts()[1:]
lista_hash_neutros=((corpus['Hash'][corpus['airline_sentiment'] == 'neutral']).apply(', '.join)).value_counts()[1:]


#crosstab de #tags com sentimentos
corpus['bins_num_Hash']=pd.cut(x=corpus['num_Hash'], bins=[-1,0,1,9] , labels=['no #', '1', '2+'])
table9 = pd.crosstab(index=corpus['bins_num_Hash'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table9
table9.plot(kind="bar", figsize=(8,8), stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

# Agrupar em lista
HT_negative_sum = sum(HT_negative,[])
HT_neutral_sum = sum(HT_neutral,[])
HT_positive_sum = sum(HT_positive,[])


a = nltk.FreqDist(HT_negative_sum)
d = pd.DataFrame({'Hashtag Negative': list(a.keys()),'Count': list(a.values())})
# Top 10 most frequent hashtags  - Negative    
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag Negative", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

a = nltk.FreqDist(HT_positive_sum)
d = pd.DataFrame({'Hashtag Positive': list(a.keys()),
                  'Count': list(a.values())})
# Top 10 most frequent hashtags  - Positive    
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag Positive", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

a = nltk.FreqDist(HT_neutral_sum)
d = pd.DataFrame({'Hashtag Neutral': list(a.keys()),
                  'Count': list(a.values())})
# Top 10 most frequent hashtags  - Neutral    
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag Neutral", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

#Nº de Hashtags Positivos
len(HT_negative)

#Nº de Hashtags Neutros
len(HT_neutral)

#Nº de Hashtags Negativos
len(HT_positive)


######### Nova Feature: Exc

#Função de contagem de caracteres
def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    return counter

# Nº de ! de negative tweets
exc_negative=list(map(lambda txt: count_occurences("!", txt),corpus['text'][corpus['airline_sentiment']=='negative']))

# Nº de ! de neutral tweets
exc_neutral=list(map(lambda txt: count_occurences("!", txt),corpus['text'][corpus['airline_sentiment']=='neutral']))

# Nº de ! de positive tweets
exc_positive=list(map(lambda txt: count_occurences("!", txt),corpus['text'][corpus['airline_sentiment']=='positive']))

#Escreve no dataset
corpus.loc[corpus.loc[corpus.airline_sentiment=='negative'].index, 'Exc'] = exc_negative
corpus.loc[corpus.loc[corpus.airline_sentiment=='neutral'].index, 'Exc'] = exc_neutral
corpus.loc[corpus.loc[corpus.airline_sentiment=='positive'].index, 'Exc'] = exc_positive
corpus['Exc']
# Soma valores individuais
exc_negative_sum = sum(exc_negative)
exc_neutral_sum = sum(exc_neutral)
exc_positive_sum = sum(exc_positive)

#Cross tab !
pd.crosstab(index=corpus['Exc'][corpus['Exc']!=0], columns=corpus['airline_sentiment']).apply(lambda r: r, axis=0)
#crosstab de bins
corpus['binsExc']=pd.cut(x=corpus['Exc'], bins=[-1,0,1,2,3,4,5,27], labels=['0', '1', '2','3','4','5','6+'])
table10 = pd.crosstab(index=corpus['binsExc'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table10


############# Nova Feature: Int

# Nº de ? de negative tweets
int_negative=list(map(lambda txt: count_occurences("?", txt),corpus['text'][corpus['airline_sentiment']=='negative']))
# Nº de ! de neutral tweets
int_neutral=list(map(lambda txt: count_occurences("?", txt),corpus['text'][corpus['airline_sentiment']=='neutral']))
# Nº de ! de positive tweets
int_positive=list(map(lambda txt: count_occurences("?", txt),corpus['text'][corpus['airline_sentiment']=='positive']))

#Escreve no dataset
corpus.loc[corpus.loc[corpus.airline_sentiment=='negative'].index, 'Int'] = int_negative
corpus.loc[corpus.loc[corpus.airline_sentiment=='neutral'].index, 'Int'] = int_neutral
corpus.loc[corpus.loc[corpus.airline_sentiment=='positive'].index, 'Int'] = int_positive

# Soma valores individuais
int_negative_sum = sum(int_negative)
int_neutral_sum = sum(int_neutral)
int_positive_sum = sum(int_positive)

#Cross tab ?
pd.crosstab(index=corpus['Int'], columns=corpus['airline_sentiment']).apply(lambda r: r, axis=0)
#crosstab de bins
corpus['binsInt']=pd.cut(x=corpus['Int'], bins=[-1,0,1,2,11], labels=['0', '1', '2','3+'])
table11 = pd.crosstab(index=corpus['binsInt'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table11

############# Nova Feature: IE

corpus['IE']=corpus['Exc']+corpus['Int']
#Cross tab 
pd.crosstab(index=corpus['IE'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
#crosstab de bins
corpus['binsIE']=pd.cut(x=corpus['IE'], bins=[-1,0,1,2,11], labels=['0', '1', '2','3+'])
table12 = pd.crosstab(index=corpus['binsIE'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table12

############# Nova Feature: Upper

# Nº palavras upper case e com mais do que 3 letras
corpus['Upper']=corpus['text'].apply(lambda x: len([x for x in x.split() if (x.isupper())]))

tb1=pd.crosstab(index=corpus['Upper'][(corpus['Upper']!=0)&(corpus['Upper']!=1)&(corpus['Upper']!=2)&(corpus['Upper']!=3)],columns=corpus['airline_sentiment']).apply(lambda r: r, axis=0)
tb1
corpus['binsUpper']=pd.cut(x=corpus['Upper'], bins=[3,4,5,6,7,8,9,26], labels=['4','5','6','7','8','9','10+'])
table13 = pd.crosstab(index=corpus['binsUpper'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table13
















#Remove os @handles
np.vectorize(remove_pattern)(corpus['text'], "@[\w]*")

















plt.figure(figsize=(5,6))
sns.countplot(x='negativereason', data=corpus['negativereason'], palette = "Set1")
plt.grid(0)
plt.xticks(rotation = 90, fontsize=12)
plt.title("Customer negative sentiment topics", fontsize=14)
plt.tight_layout()







corpus['retweet_count'].hist(bins=[1,2,44], figsize=(15, 15));
plt.show()



#crosstab por sentimento e companhia
table6 = pd.crosstab(index=corpus['bins_len_text'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table1 = pd.crosstab(index=corpus['bins_retweet'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)
table1





#histogramas: não mostra nada de jeito
numerical=subset.select_dtypes(include=[np.number]).columns.tolist()
subset[numerical].distplot(bins=10, figsize=(15, 15), layout=(3, 2));
plt.show()





#Pairplot mas nao se vê grande coisa
corpus['dumm_retweet_count']=pd.cut(x=corpus['retweet_count'], bins=[-1,0,44], labels=[0,1])
cols=['airline_sentiment_confidence','dumm_retweet_count', 'len_text', 'airline_sentiment']
g = sns.pairplot(corpus[cols], hue="airline_sentiment")
plt.show()

cols=[ 'len_text', 'airline_sentiment']
sns.distplot(corpus[cols])
plt.show()






sns.distplot(data)
plt.show()

cols=[ 'len_text', 'airline_sentiment']
g = sns.pairplot(corpus[cols], hue="airline_sentiment")
plt.show()

data.plot(kind="kde")
plt.show()



#%% 1- Funcoes para processar "@name"
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt   

regex = r"@(VirginAmerica|united|SouthwestAir|Delta|USAirways|AmericanAir)"
def text_replace(text):
    return re.sub(regex, '@airlinename', text, flags=re.IGNORECASE)

corpus['text_airlinename'] = corpus['text'].apply(text_replace)

np.vectorize(remove_pattern)(corpus['text'], "@[\w]*")

#Nova feature - Nº Ocorrencias do nome da companhia aérea
corpus['num_oc']=corpus.text_airlinename.apply(lambda x: x.count('@airlinename'))
corpus['bins_num_oc']=pd.cut(x=corpus['num_oc'], bins=[-1,0,1,5], labels=['0','1','2+'])
corpus['num_oc'].value_counts()
pd.crosstab(index=corpus['bins_num_oc'], columns=corpus['airline_sentiment']).apply(lambda r: r/r.sum(), axis=0)


#%%

################### DATA PREPARATION #######################

