#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tqdm')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[3]:


df = pd.read_csv('twitter_data.csv')
df.head(3)


# In[4]:


print(df.shape)


# In[5]:


print(df.isnull().sum())


# In[6]:


ax = df['category'].value_counts().sort_index().plot(kind='bar', 
                                                     title='Count of Twitter Reviews by Category', 
                                                     figsize=(10, 5))
ax.set_xlabel('Twitter Review Sentiment Score')
plt.show()


# ## VADER 

# In[8]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.std import tqdm # to show progress bar

sia = SentimentIntensityAnalyzer()


# In[9]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['clean_text']
        myid = row['id']

        # Noticed clean_text has some null values
        if pd.isna(text):
            text = ""

        text = str(text)
        
        res[myid] = sia.polarity_scores(text)
        
    except Exception as e:
        print(f'Error for id {myid}: {e}')


# In[10]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'id'})
vaders = vaders.merge(df, how='right')

vaders = vaders.rename(columns={'pos': 'vader_pos', 'neu': 'vader_neu', 'neg': 'vader_neg'})

vaders.head(3)


# In[11]:


# overall plot
ax = sns.barplot(data=vaders, x='category', y='compound')
ax.set_title('Compound Score by Twitter Review')
plt.show()


# In[12]:


# specific plots
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='category', y='vader_pos', ax=axs[0])
sns.barplot(data=vaders, x='category', y='vader_neu', ax=axs[1])
sns.barplot(data=vaders, x='category', y='vader_neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# ## Roberta Model

# In[14]:


get_ipython().system('pip install transformers scipy')


# In[15]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[16]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[17]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" # using trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[18]:


# roberta function
def polarity_scores_roberta(example):
    if pd.isna(example):
        example = ""
    example = str(example)
    
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    
    return scores_dict


# In[19]:


get_ipython().system('pip install swifter')


# In[20]:


import swifter # improve execution time

tqdm.pandas()

# Convert vaders DataFrame into a dictionary for fast lookup
vaders_dict = vaders.set_index('id')[['vader_neg', 'vader_neu', 'vader_pos']].to_dict('index')

# Efficiently map VADER scores to df using .map()
df['vader_neg'] = df['id'].map(lambda x: vaders_dict.get(x, {}).get('vader_neg', None))
df['vader_neu'] = df['id'].map(lambda x: vaders_dict.get(x, {}).get('vader_neu', None))
df['vader_pos'] = df['id'].map(lambda x: vaders_dict.get(x, {}).get('vader_pos', None))

# Ensure clean_text is a string to avoid errors
df['clean_text'] = df['clean_text'].astype(str)

# Apply RoBERTa function in parallel using swifter and tqdm
df[['roberta_neg', 'roberta_neu', 'roberta_pos']] = df['clean_text'].progress_apply(
    lambda x: pd.Series(polarity_scores_roberta(x))
)

df.to_csv("twitter_sentiment_outputs.csv", index=False)
print("CSV file saved successfully!")
print(df.head(3))


# In[21]:


# Compare vader & roberta 
sns.pairplot(data = df, 
             vars = ['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'], 
             hue = 'category', 
             palette = 'tab10')
plt.show()


# ## Output Validation

# ### Negative category, Positive score (VADER & ROBERTA)

# In[24]:


df.query('category == -1').sort_values('roberta_pos', ascending=False)['clean_text'].values[0]


# In[25]:


df.query('category == -1').sort_values('vader_pos', ascending=False)['clean_text'].values[0]


# ### Positive category, Negative score (VADER & ROBERTA)

# In[27]:


df.query('category == 1').sort_values('roberta_neg', ascending=False)['clean_text'].values[0]


# In[28]:


df.query('category == 1').sort_values('vader_neg', ascending=False)['clean_text'].values[0]


# # Model Evaluation

# In[53]:


# store compound score from VADER
compound_df = vaders[['id', 'compound']].rename(columns={'compound': 'vader_compound'})

# Merge with the original df (assuming 'id' is the common key)
df = df.merge(compound_df, on='id', how='left')

# Save the updated dataframe with compound score
df.to_csv("twitter_sentiment.csv", index=False)
print("CSV file updated successfully with VADER compound scores!")
print(df.head(3))


# In[65]:


# create labels using VADER compound
def vader_sentiment_label(compound):
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_label"] = df["vader_compound"].apply(vader_sentiment_label)


# In[67]:


# create labels using Roberta
def roberta_sentiment_label(row):
    scores = {"negative": row["roberta_neg"], 
              "neutral": row["roberta_neu"], 
              "positive": row["roberta_pos"]}
    return max(scores, key=scores.get)  # Get the label with the highest probability

df["roberta_label"] = df.apply(roberta_sentiment_label, axis=1)


# In[71]:


# measure agreements between vader and Roberta
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(df["vader_label"], df["roberta_label"])

print("Agreement between VADER and RoBERTa (Cohen’s Kappa):", kappa)


# In[73]:


# measure disagreements
disagreements = df[df["vader_label"] != df["roberta_label"]]

print(disagreements[["clean_text", 
                     "vader_label", 
                     "roberta_label", 
                     "vader_compound", 
                     "roberta_neg", 
                     "roberta_neu", 
                     "roberta_pos"]].head(10))


# In[75]:


# visualize difference
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(df["vader_label"], 
                      df["roberta_label"], 
                      labels=["negative", "neutral", "positive"])

sns.heatmap(cm, annot=True, 
            fmt='d', 
            cmap="Blues", 
            xticklabels=["Negative", "Neutral", "Positive"], 
            yticklabels=["Negative", "Neutral", "Positive"])

plt.xlabel("Predicted by RoBERTa")
plt.ylabel("Predicted by VADER")
plt.title("VADER vs RoBERTa Confusion Matrix")
plt.show()

