---
title: "Sentiment Analysis"
date: 2020-01-13
tag: [NLP]
---

# Sentiment Analysis of a text in the email using NLTK.

Importing the libraries

```python
import pandas as pd
import numpy as np
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

Loading the dataset
```python
df = pd.read_csv("/Users/vamshi/Desktop/SMSSpamCollection.tsv",sep = "\t",header = None)
df.head()
```
0	1
0	ham	I've been searching for the right words to tha...
1	spam	Free entry in 2 a wkly comp to win FA Cup fina...
2	ham	Nah I don't think he goes to usf, he lives aro...
3	ham	Even my brother is not like to speak with me. ...
4	ham	I HAVE A DATE ON SUNDAY WITH WILL!!

Data preprocessing
```python
cols = ["Response","Mail"]
df.columns = cols
df.head()
Data preprocessing
```
Response	Mail
0	ham	I've been searching for the right words to tha...
1	spam	Free entry in 2 a wkly comp to win FA Cup fina...
2	ham	Nah I don't think he goes to usf, he lives aro...
3	ham	Even my brother is not like to speak with me. ...
4	ham	I HAVE A DATE ON SUNDAY WITH WILL!!

```python
df.isnull().sum()
df["Response"].value_counts()
```
Finding blank or empty fields
```python
blanks =[]
for i,lb,rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blank.append(i)
```
```python

def sentiment_analysis(text):
    score = senti.polarity_scores(text)
    if score["compound"] == 0:
        return "Netural"
    elif score["compound"] >0:
        return "Positive"

    else:
        return "Negative"
```
Adding scores to the dataframe
```python
df['scores']= df['Mail'].apply(lambda Mail:senti.polarity_scores(Mail))
df['compound_score']= df['scores'].apply(lambda k:k['compound'])
df['sentiment'] = df['compound_score'].apply(lambda compound_score: "Positive" if compound_score >= 0 else "Negative")
```
Final Result:

Response	            Mail	                                         Compound_Score	      Sentiment
0	       ham	I've been searching for the right words to tha...	 	    0.9143	             Positive
1	       spam	Free entry in 2 a wkly comp to win FA Cup fina...		    0.7964	             Positive
2	       ham	Nah I don't think he goes to usf, he lives aro...		   -0.1027               Negative
3	       ham	Even my brother is not like to speak with me. ...	     -0.1331	             Negative
4	       ham	I HAVE A DATE ON SUNDAY WITH WILL!!	{'neg': 0.0,   	    0.0000	             Positive
