# Ticket-Assignment-System

# Development of an Intelligent Support Ticket Assignment System Using Machine Learning

## Description
This notebook focuses on building a machine learning-based support ticket assignment system to classify and assign customer complaints to specific topics or departments for effective resolution.

## Setup
1. **Import Libraries**:
   - `json`, `numpy`, `pandas` for data handling.
   - `matplotlib`, `seaborn`, `plotly` for visualizations.
   - `sklearn` for machine learning models and evaluation.
   - `spacy` and `nltk` for natural language processing.

2. **Google Drive Mount**:
   If working on Google Colab, ensure the dataset is accessible by mounting Google Drive.

3. **Load Dataset**:
   Load the JSON file into a Python dictionary and normalize it to a DataFrame using `pd.json_normalize`.

## Data Preparation
1. **Column Renaming and Merging**:
   - Rename columns to more meaningful names.
   - Merge categories for easier topic assignment.
   
2. **Missing Values**:
   - Replace empty strings with NaN and remove rows with NaN values in the complaints.

3. **Text Cleaning and Lemmatization**:
   - Write a function to clean the text by removing unnecessary elements.
   - Lemmatize text to convert words to their base form.

4. **POS Tag Extraction**:
   - Extract noun phrases from the text to create a clean dataset.

## Exploratory Data Analysis (EDA)
1. **Character Length Distribution**:
   - Plot a histogram to analyze the length distribution of complaints.

2. **Word Cloud**:
   - Generate a word cloud to visualize the most frequently used words.

3. **N-gram Analysis**:
   - Analyze and visualize the top 30 unigrams, bigrams, and trigrams.

4. **Class Imbalance**:
   - Check for any class imbalance in the topics.

## Feature Extraction and Topic Modeling
1. **TF-IDF Vectorization**:
   - Initialize a `TfidfVectorizer` and create a Document-Term Matrix.

2. **NMF Topic Modeling**:
   - Fit an NMF model to the document-term matrix and identify significant topics.

3. **Assign Dominant Topic**:
   - Create a column for the dominant topic in each complaint and map topics to more readable names.

## Model Training and Evaluation
1. **Prepare Training Data**:
   - Split data into training and testing sets using `train_test_split`.
   - Use `CountVectorizer` and `TfidfTransformer` for vectorization and transformation.

2. **Model Selection and Evaluation**:
   - Implement and evaluate multiple models:
     - **Multinomial Naive Bayes**
     - **Logistic Regression**
     - **Decision Tree**
     - **Random Forest**

   - Create a utility function to display the classification report, confusion matrix, and ROC AUC scores.

3. **Stratified K-Fold Cross-Validation**:
   - Use StratifiedKFold for cross-validation with 5 folds.

4. **GridSearchCV for Hyperparameter Tuning**:
   - Use `GridSearchCV` to optimize hyperparameters for each classifier and print the best parameters.

5. **Metrics Table**:
   - Create a comparison table to show performance metrics of different models.

## Complaint Prediction
- Test the best model by predicting the topics of raw complaint texts.

## Model Serialization
- Serialize the best model and TF-IDF vectorizer using `joblib` for later use.

