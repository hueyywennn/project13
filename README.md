# Twitter Sentiment Analysis
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Project Overview
This project analyzes public sentiment on Twitter using natural language processing (NLP) techniques. The goal is to classify tweets as positive, negative, or neutral, providing insights into public opinion on a specific topic, brand, or event.

### Dataset 
**Dataset Source:** [Twitter Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset/data)

| Name              | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| clean_text        | Tweets collected from twitter                                      |
| category          | Sentiments labels: negative(-1), neutral(0), and positive(+1)      |

## Project Objectives
1. **Data Preprocessing**: Cleaning and preparing text data for analysis.
2. **Exploratory Data Analysis (EDA)**: Understanding sentiment distribution and patterns in tweets.
3. **Feature Engineering**: Extracting relevant textual features for sentiment classification.
4. **Model Training & Evaluation**: Implementing different NLP models to classify tweets and comparing their performance.
5. **Visualization & Insights**: Presenting findings through data visualizations and sentiment comparisons.

## Machine Learning Models Used
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**  
  - A rule-based sentiment analysis tool that assigns sentiment scores based on a predefined lexicon.
- **RoBERTa (Robustly Optimized BERT Pretraining Approach)**  
  - A transformer-based deep learning model that leverages contextual embeddings for more accurate sentiment classification.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, transformers, torch, NLTK, vaderSentiment, swifter, scipy
- **Data Processing**: NLTK for text preprocessing, VaderSentiment for rule-based sentiment analysis
- **Visualization Tools**: seaborn, matplotlib for graphical analysis

## Project Workflow
1. **Data Collection**: Import and inspect the dataset.
2. **Data Cleaning & Preprocessing**: Remove noise, tokenize text, and prepare data for analysis.
3. **Exploratory Data Analysis (EDA)**: Visualize sentiment distributions using bar plots.
4. **Feature Engineering**: Extract meaningful textual features for better classification.
5. **Model Training**: Train VADER and RoBERTa models for sentiment classification.
6. **Model Evaluation**: Compare classification performance using Cohen’s Kappa Score and Confusion Matrix.
7. **Results Interpretation**: Analyze and present insights from the sentiment classification models.
