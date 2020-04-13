---
layout: archive
permalink: /projects/
title: "Anime Recommendation"
author_profile: true
header:
  image: "/images/image2.jpg"
mathjax: "true"

---

{% include base_path %}
{% include group-by-array
collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}


Created a recommendation system using K-Nearest Neighbor and other algorithms where a user can be recommended various shows and genres based on the user rating and user count.


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
