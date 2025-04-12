# CODTECH Internship - SENTIMENT-ANALYSIS Tasks

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: ANUBHAV SINGH

*INTERN ID*: CT06DA708

*DOMAIN*: DATA ANALYTICS

*DURATION*: 6 WEEEKS

*MENTOR*: NEELA SANTOSH


## Task 4: IMDB Movie Review Sentiment Analysis

### Overview

This project performs sentiment analysis on the IMDB movie review dataset. Using Natural Language Processing (NLP) techniques and a Naive Bayes Classifier, the project aims to classify movie reviews as either positive or negative based on their textual content. This showcases sentiment analysis and text classification using machine learning.

### Tasks Performed

*   Installed required libraries for NLP and machine learning (nltk, scikit-learn, matplotlib, seaborn).
*   Downloaded the IMDB movie review dataset.
*   Loaded the training dataset, separating review text and sentiment labels.
*   **Text Preprocessing:**
    *   Removed HTML tags from the review text.
    *   Removed non-alphabetic characters.
    *   Converted text to lowercase and split into words.
    *   Applied stemming using PorterStemmer to reduce words to their root form.
    *   Removed stop words (common English words) to focus on meaningful content.
*   **Feature Extraction:**
    *   Used CountVectorizer to convert the preprocessed text reviews into numerical feature vectors (Bag-of-Words representation), limiting to the top 5000 features.
*   Split the feature vectors and labels into training and testing sets.
*   Trained a Multinomial Naive Bayes Classifier on the training data.
*   Predicted sentiment labels for the test data.
*   **Model Evaluation:**
    *   Calculated the accuracy of the Naive Bayes classifier.
    *   Generated a classification report (precision, recall, F1-score, support).
    *   Visualized the confusion matrix using a heatmap.
*   Summarized key insights from the model evaluation.

### Code and Resources Used

*   **Programming Language:** Python
*   **Libraries:** nltk, scikit-learn, matplotlib, seaborn
*   **Dataset:** IMDB Movie Review Dataset (aclImdb)
*   **Notebooks:**
    *   [Link to your Task 4 Notebook (task4.ipynb)](task4.ipynb)

### Results and Insights

*   **Model Accuracy:** The Naive Bayes classifier achieved an accuracy of approximately 84.6% on the test set.
*   **Classification Report:**  The classification report shows balanced precision, recall, and F1-score for both positive (1) and negative (0) sentiment classes, indicating good performance for both sentiment types.
*   **Confusion Matrix:** The confusion matrix visually confirms the model's performance, showing a relatively small number of misclassifications.
*   **Suggestions:**  Further improvements could be explored by using TF-IDF vectorization instead of CountVectorizer or by experimenting with more complex deep learning models for sentiment analysis.

### How to Run (Optional)

The code is designed to be run in Google Colab. Simply open the `task4.ipynb` notebook in Colab and run the cells sequentially. Make sure to download the NLTK stopwords data when prompted during execution.

