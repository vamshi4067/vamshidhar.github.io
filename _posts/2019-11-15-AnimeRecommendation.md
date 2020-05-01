---
title: "Anime Recommendation"
date: 2019-08-15

---

# Created a recommendation system using K-Nearest Neighbor.

Recommendation system using K-Nearest Neighbor and other algorithms where a user can be recommended various shows and genres based on the user rating and user count.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
anime = pd.read_csv('/Users/vamshi/Library/Mobile Documents/com~apple~CloudDocs/anime.csv')
```

Handling null values and Converting datatype of episodes from object to numeric
```python
anime.isna().sum()
anime['episodes'] = anime['episodes'].replace('Unknown',np.nan)
anime['episodes'] = pd.to_numeric(anime['episodes'])
anime['episodes'].dtype
rating.rename(columns={"rating":"User_rating"},inplace=True)
rating['User_rating'] = rating['User_rating'].replace(-1,np.nan)
rating = rating.dropna()
```

Defining the definition of like by calculating mean of rating
```python
mrpu = rating.groupby('user_id').mean().reset_index()
mrpu.drop('anime_id',axis=1,inplace=True)
mrpu.rename(columns={'User_rating':'Mean_rating'},inplace=True)
cpu = rating.groupby('user_id').count().reset_index()
cpu.drop('anime_id',axis=1,inplace=True)
cpu.rename(columns={'User_rating':'Num_mvs_rated'},inplace=True)
mrpu = pd.merge(mrpu,cpu,on=['user_id','user_id'])
rating = pd.merge(rating,mrpu,on=['user_id','user_id'])
```

Removing the shows the user didn't like and having num of movies rated <5 and combining the two datasets of anime and ratings
and considering only 10000 user data.
```python
rating = rating.drop(rating[rating['User_rating']<rating['Mean_rating']].index)
total_data = pd.merge(anime,rating,on=['anime_id','anime_id'])
total_data = total_data[total_data['user_id']<=10000]
```

Finding total users who gave ratings to the movie and finding average user rating given by the users per each movie
```python
temp = most_liked.groupby('name').count().reset_index()
temp = temp.rename(columns={"User_rating":"Num_ratingmvie"})
temp1 = most_liked.groupby('name').mean().reset_index()
temp1 = temp1.rename(columns={"User_rating":"Mean_rating_permovie"})
most_liked = pd.merge(temp,temp1,on=['name','name'])
most_liked = most_liked.sort_values(['Num_ratingmvie','Mean_rating_permovie'],ascending=[False,False])
```

Type of show mostly liked and the optimal number of clusters are 5
```python
type_liked = total_data.groupby('type').count()['Num_mvs_rated'].reset_index()
type_liked = type_liked.rename(columns={'Num_mvs_rated':'Freq_of_animetype'})
best_model = KMeans(n_clusters=5)
best_model.fit(total_data.iloc[:,[0,4,5,6,7,8,9]])
predictions = best_model.predict(total_data.iloc[:,[0,4,5,6,7,8,9]])
total_data['cluster'] = predictions
total_data[total_data['cluster']==4]
```
```python
from wordcloud import WordCloud

def makeCloud(Dict,name,color):
    words = dict()

    for s in Dict:
        words[s[0]] = s[1]

        wordcloud = WordCloud(
                      width=1500,
                      height=500,
                      background_color=color,
                      max_words=20,
                      max_font_size=500,
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)


    fig = plt.figure(figsize=(12, 8))
    plt.title(name)
    plt.imshow(wordcloud)
    plt.axis('off')

    plt.show()
In [119]:
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split(','):        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]:
            if pd.notnull(s): keyword_count[s] += 1

    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count
In [120]:
set_keywords = set()
for liste_keywords in total_data['genre'].str.split(',').values:
    if isinstance(liste_keywords, float): continue  
    set_keywords = set_keywords.union(liste_keywords)
Favorite genre for these clusters

In [130]:
#cluster 1
keyword_occurences, dum = count_word(total_data[total_data['cluster']==0], 'genre', set_keywords)
makeCloud(keyword_occurences[0:10],"cluster 1","lemonchiffon")

Given the name of the show predicting some shows for recommending to the user
```python
def predict_shows(show_name):
    cluster_num = total_data[total_data['name'] == show_name]['cluster'].unique()[0]
    avg_userrating = total_data[total_data['name'] == show_name]['User_rating'].mean()
    recom = total_data[total_data['cluster']==cluster_num]
    recom = recom[recom['name']!=show_name]
    recom = recom[(recom['User_rating']<avg_userrating+0.5) & (recom['User_rating']>=avg_userrating-0.5)]
    recom = recom.groupby('name').mean().reset_index()
    recom = recom.drop(['anime_id','user_id','cluster'],axis=1)
    recom = recom.rename(columns={'User_rating':'Meanuserrating/anime','Mean_rating':'Avg(Mean_rating)/anime','Num_mvs_rated':'Avg(Num_mvs_rated)/anime'})
    recom = recom.sort_values(['Meanuserrating/anime','Avg(Mean_rating)/anime','Avg(Num_mvs_rated)/anime'],ascending=[False,True,False])
    print(recom['name'][0:15])
```

We could determine which anime is liked by most of the users and where able to find the similar anime for recommending to the users and to find which is the most liked genre by the users.We even found which type of show is mostly liked by the users.
Finally we could predict what are the recommended shows given a user-id or show name.
