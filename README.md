---
title: Customer Support Ticket Classifier
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
---

# Customer Support Ticket Classification & Entity Extraction

### 🚀 [Live Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/nishananzr/customer-ticket-classifier-demo)

## Project Overview

This project presents a complete machine learning pipeline designed to automate the initial processing of customer support tickets. The system takes raw ticket text as input and performs three key tasks:
1.  **Classifies the ticket by its `issue_type`** (e.g., Billing Problem, Product Defect).
2.  **Predicts the `urgency_level`** of the ticket (Low, Medium, High).
3.  **Extracts key entities** from the text, such as product names, dates, and complaint-related phrases.

The goal is to streamline the customer support workflow by automatically routing tickets to the correct department with a pre-analyzed summary, significantly reducing manual effort and response times. This solution was built using traditional NLP and classical machine learning models.

## My Approach & Project Workflow

The project was developed following a structured machine learning workflow, focusing on robust data analysis, thoughtful feature engineering, and critical model evaluation.

### 1. Data Preparation & Exploratory Data Analysis (EDA)
- **Initial Inspection & Data Cleaning:** The dataset was loaded and thoroughly inspected for shape, data types, and missing values. It was found that several critical columns, including `ticket_text`, `issue_type`, and `urgency_level`, had missing entries. **My Design Choice:** To ensure model quality and avoid introducing noise through imputation, I made the decision to drop any row with missing data in these critical columns. This resulted in a clean, complete dataset of 826 records for training.
- **Text Preprocessing:** A comprehensive text preprocessing pipeline was created to normalize the ticket text. This involved:
    - Lowercasing all text.
    - Removing punctuation and special characters using regular expressions.
    - **Tokenization:** Splitting text into individual words.
    - **Stopword Removal:** Eliminating common English words (e.g., "the", "a", "is") that provide little predictive value.
    - **Lemmatization:** Reducing words to their root form (e.g., "running" -> "run") to consolidate the vocabulary using NLTK's `WordNetLemmatizer`.
    - **Other preprocessing steps tried:** Also tried to detect type errors in the query text , but the text may contain the product/brand name and since machine will learn like these unique names as type errors, this cleaning step was eliminated.


### 2. Feature Engineering
The core of this project was converting the cleaned text into meaningful numerical features.
- **TF-IDF Vectorization (Primary Feature):** I chose **TF-IDF (Term Frequency-Inverse Document Frequency)** as my primary vectorization technique. **My Design Choice:** This was selected over a simpler Bag-of-Words model because TF-IDF provides a more nuanced weighting. It down-weights words common across all tickets (e.g., "customer") and gives more importance to words specific to certain ticket types (e.g., "broken," "refund"), providing a much stronger signal for classification. I also included bi-grams (`ngram_range=(1, 2)`) to capture the context of two-word phrases like "late delivery".
- **Additional Features:** To enrich the feature set, I engineered two more features:
    1.  **Ticket Length:** The word count of a ticket can indicate its complexity.
    2.  **Sentiment Polarity:** The emotional tone of the ticket (from -1.0 for negative to +1.0 for positive), extracted using `TextBlob`, serves as a strong potential indicator for urgency.
- **Feature Combination:** The sparse TF-IDF matrix was combined with these two new numerical features (which were scaled using `StandardScaler`) into a single, unified feature matrix for model training.

### 3. Multi-Task Learning: Two Separate Models
Two distinct models were built to handle the separate prediction tasks as required.

1.  **Issue Type Classifier:** A `LogisticRegression` model was trained to predict the `issue_type`. This model is fast, interpretable, and provides a strong baseline.
2.  **Urgency Level Classifier:** A `RandomForestClassifier` was trained to predict the `urgency_level`. This model was chosen for its ability to capture more complex, non-linear relationships in the data.

### 4. Entity Extraction
A rule-based approach was implemented to extract key entities. **My Design Choice:** I initially developed a simple keyword-based extractor but found it inflexible. I then implemented a more advanced and robust method using **NLTK's Part-of-Speech (POS) tagging and chunking**. This allows the system to identify:
- **Products:** By matching known noun phrases against a product list.
- **Dates:** Using a comprehensive regex pattern.
- **Complaint Phrases:** By identifying and extracting verb phrases associated with common complaint root words (e.g., "stopped working", "is lost"). This is more powerful than matching single keywords.

## Model Evaluation & Critical Findings

This is the most crucial part of the analysis. The performance of the two models tells very different stories about the dataset and demonstrates the importance of critical evaluation.

### Model 1: Issue Type Classifier Evaluation

The `LogisticRegression` model achieved a **perfect 1.0 F1-score** on the test set.

                precision    recall  f1-score   support

Account Access       1.00      1.00      1.00        27

      accuracy                           1.00       166
     macro avg       1.00      1.00      1.00       166
  weighted avg       1.00      1.00      1.00       166

  
**Challenge & Critical Finding: Data Leakage Investigation**
A perfect score is a major red flag for data leakage. I performed a diagnostic test which confirmed that **120 rows (~14% of the dataset) contained their own label** within the ticket text (e.g., a ticket with the text "I have an installation issue" was labeled `Installation Issue`).

This leakage is the reason for the perfect score. The model is correctly learning the direct patterns present *in this specific dataset*. This finding is a critical insight into the data's quality and demonstrates my ability to diagnose unexpected model behavior.

### Model 2: Urgency Level Classifier Evaluation

The `RandomForestClassifier` achieved a **low weighted F1-score of 0.35**.

          precision    recall  f1-score   support

    High       0.41      0.36      0.39        58
     Low       0.34      0.40      0.37        52
  Medium       0.30      0.29      0.29        56

accuracy                           0.35       166


**Challenge & Analysis:**
The low score indicates that the engineered features are **insufficient** to predict urgency. To investigate, I tried a simpler `LogisticRegression` model (which also performed poorly) and conducted a feature importance analysis. The analysis showed the model was not picking up on strong, intuitive "urgent" keywords.

**Conclusion:** Ticket urgency in this dataset is not determined by simple keywords or basic sentiment. The signal is likely hidden in more complex semantic structures, or the labels themselves are inherently noisy. This analysis demonstrates a realistic and challenging classification task where initial feature engineering is not enough, pointing the way for future iterative development.

## Project Limitations

- **Urgency Model Performance:** The urgency classifier's performance is too low for production use and highlights the difficulty of the task with the given data.
- **Entity Extraction:** The rule-based entity extraction is not robust to new product names or unconventional complaint phrasing that falls outside the defined grammatical patterns.
- **Data Leakage:** The `issue_type` model's perfect score is a result of data leakage and its real-world performance would be much lower on a clean dataset.

## Setup and Installation Instructions

To run this project locally, please follow these steps:

1.  **Prerequisites:**
    - Python 3.8+
    - Git and Git LFS installed (`git lfs install`).

2.  **Clone the Repository:**
    ```bash
    git clone https://huggingface.co/spaces/nishananzr/customer-ticket-classifier-demo
    cd customer-ticket-classifier-demo
    ```

3.  **Download the Dataset:**
    The data file is not included in the repository. Please download it from the link below and place it in the `data/` directory.
    - **[Download ai_dev_assignment_tickets_complex_1000.xlsx]**
    - ([https://docs.google.com/spreadsheets/d/1wQeTvEqI4TeQ3c-v8bEEuAc8-oWM1A0K/edit?usp=sharing&ouid=100728351340949217592&rtpof=true&sd=true])

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Analysis Notebook:**
    To see the full data analysis and model training process, run the Jupyter Notebook:
    ```bash
    jupyter notebook ticket_classification_analysis.ipynb
    ```

6.  **Launch the Interactive App:**
    To launch the Gradio web interface, run the following command in your terminal:
    ```bash
    python app.py
    ```
    Then, open your web browser to `http://127.0.0.1:7860`.

