Model build on the Random Forest algorithm. Analyzes news articles about different cryptocurrencies and categorize them as potentially connected with crime activity (class 1) or not (class 0).
Model was trained on 3 datasets (about 58 000 news together):
1) crypto_news_parsed_2013-2017_train.csv (https://www.kaggle.com/datasets/kashnitsky/news-about-major-cryptocurrencies-20132018-40k);
2) crypto_news_parsed_2018_validation.csv (https://www.kaggle.com/datasets/kashnitsky/news-about-major-cryptocurrencies-20132018-40k);
3) cryptonews.csv (https://www.kaggle.com/datasets/oliviervha/crypto-news).  
Content:
- datasets_preparation.py - cleaning, concatenating and labeling of the data;
- keywords.txt - for labeling the data;
- model_training.py - train a model with prepared data;
- model_using.py - get predictions on a new data.
Accuracy - 85%.
