# Sentiment Analysis for Mental Health
Mental health is more than the absence of mental disorders. It exists on a complex continuum, which is experienced differently from one person to the next, with varying degrees of difficulty and distress and potentially very different social and clinical outcomes. 

Mental health conditions include mental disorders and psychosocial disabilities as well as other mental states associated with significant distress, impairment in functioning, or risk of self-harm. People with mental health conditions are more likely to experience lower levels of mental well-being, but this is not always or necessarily the case.

### Determinants of Mental Health 
- **Mental Health Continuum**: Mental health is influenced by multiple factors that either protect or undermine it, shifting individuals along a continuum throughout their lives.
  
- **Individual Risk Factors**:
  - Psychological and biological factors (e.g., emotional skills, substance use, genetics) increase vulnerability to mental health problems.
  - Developmentally sensitive periods, especially early childhood, are critical; adverse experiences like harsh parenting, physical punishment, or bullying can be particularly harmful.

- **Social and Structural Risk Factors**:
  - Unfavorable social, economic, geopolitical, and environmental conditions (e.g., poverty, violence, inequality, environmental deprivation) elevate the risk of mental health issues.
  - Global threats such as economic downturns, disease outbreaks, humanitarian crises, and climate change pose risks to entire populations.

- **Protective Factors**:
  - Strengthen resilience through individual attributes (e.g., social and emotional skills) and positive external influences (e.g., quality education, safe neighborhoods, community cohesion).
  - These factors occur at all life stages and help mitigate risks.

- **Limited Predictability**:
  - No single risk or protective factor strongly predicts mental health outcomes.
  - Many people exposed to risk factors do not develop mental health conditions, while others with no known risks may still experience them.

- **Interacting Determinants**: The combination of various factors ultimately enhances or undermines mental health across individuals and populations. 

## Features of Project

The solution involves data exploration, cleaning, and preprocessing, followed by building predictive models using Decision Tree and Neural Network algorithms.

- **Data Cleaning**: Handling missing values through imputation.
- **Binary Classification**: Transforming multi-class labels into binary categories (normal/depression).
- **Model Development**: Implementing Decision Tree and Neural Network models for sentiment classification.
- **Evaluation**: Assessing model performance using accuracy, precision, recall, and ROC analysis.
- **Testing**: Validating the models with new test statements.

### Includes 
- **Python scripts**: [Solution](https://github.com/syahmishamz/Data-Analytics-ML/blob/main/Solutions/ITS69304_Group1_GroupAssignment.ipynb)
- **Detailed report**: 
- **Presentation video**: 

## About Dataset 
[**mentalhealth.csv**](https://github.com/syahmishamz/Data-Analytics-ML/blob/main/mentalhealth.csv): This comprehensive dataset is a meticulously curated collection of mental health statuses tagged from various statements. 

The dataset amalgamates raw data from multiple sources, cleaned and compiled to create a robust resource for developing chatbots and performing sentiment analysis.

### Data Overview
The dataset consists of statements tagged with one of the following seven mental health statuses:
- Normal
- Depression
- Suicidal
- Anxiety
- Stress
- Bi-Polar
- Personality Disorder

### Data Collection
The data is sourced from diverse platforms including social media posts, Reddit posts, Twitter posts, and more. Each entry is tagged with a specific mental health status, making it an invaluable asset for:
- Developing intelligent mental health chatbots.
- Performing in-depth sentiment analysis.
- Research and studies related to mental health trends.

### Features
- **unique_id**: A unique identifier for each entry.
- **Statement**: The textual data or post.
- **Mental Health Status**: The tagged mental health status of the statement.

### Usage
This dataset is ideal for training machine learning models aimed at understanding and predicting mental health conditions based on textual data. It can be used in various applications such as:
- Chatbot development for mental health support.
- Sentiment analysis to gauge mental health trends.
- Academic research on mental health patterns.
