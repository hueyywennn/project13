# Twitter Sentiment Analysis
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
to analyze public sentiment on Twitter using natural language processing (NLP) techniques and classify tweets as positive, negative, or neutral to gain insights into public opinion on a specific topic, brand, or event.

### Dataset 
(https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset/data)

| Name              | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| clean_text        | Tweets collected from twitter                                      |
| category          | Sentiments labels: negative(-1), neutral(0), and positive(+1)      |

### Features
1. **Exploratory Data Analysis (EDA)**
  -	Distribution Plots: bar plots
2. **Machine Learning Models**
  -	VADER (rule-based model): classified tweets as positive, negative, or neutral based on compound scores
  -	RoBERTa (transformer-based model): used pre-trained RoBERTa model for contextual sentiment classification
3. **Model Evaluation**
 	- Cohen’s Kappa Score: measured agreement between VADER and RoBERTa sentiment predictions
 	- Confusion Matrix: analyzed classification accuracy and misclassifications for each model
4. **Interactive Visualizations**
  - pairplot analysis of VADER vs. RoBERTa
