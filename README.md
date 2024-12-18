
# **Stock Movement Analysis Based on Social Media Sentiment**

A machine learning project that predicts stock movements by analyzing sentiment from Reddit discussions.

----------

## **Table of Contents**

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Setup and Installation](#setup-and-installation)
4.  [Usage](#usage)
5.  [Project Workflow](#project-workflow)
6.  [Results](#results)
7.  [Future Enhancements](#future-enhancements)
8.  [Contributing](#contributing)
9.  [License](#license)

----------

## **Introduction**

This project aims to predict stock price movements by analyzing sentiment from user discussions on Reddit. It combines **Natural Language Processing (NLP)** and **machine learning** to extract insights and predict trends.

----------

## **Features**

-   **Data Scraping**: Automates data collection from Reddit threads using the `PRAW` library.
-   **Sentiment Analysis**: Processes user comments to identify polarity and subjectivity using the `NLTK` library.
-   **Stock Prediction**: Machine learning model trained to predict stock trends using the extracted sentiment.
-   **Visualization**: Graphical representation of data trends and predictions using `matplotlib` and `seaborn`.

----------

## **Setup and Installation**

### **Prerequisites**

-   Python 3.8 or higher
-   Libraries:
    -   `praw`
    -   `nltk`
    -   `scikit-learn`
    -   `pandas`
    -   `numpy`
    -   `matplotlib`
    -   `seaborn`

### **Installation**

1.  Clone the repository:

    
    `git clone https://github.com/your-username/stock-movement-analysis.git
    cd stock-movement-analysis` 
    
2.  Create and activate a virtual environment:
    
   
    
    `python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate` 
    
3.  Install the required libraries:

    
    `pip install -r requirements.txt` 
    If you prefer manual installation, you can install the libraries individually:


    `pip install praw pandas nltk scikit-learn matplotlib seaborn`
    
4.  Add your Reddit API credentials to `config.py`:
    
    
    
    `client_id = "your_client_id"
    client_secret = "your_client_secret"
    user_agent = "your_user_agent"`
 #### How this works:
 1. The sensitive credentials (Reddit client ID, client secret, and user agent) are stored securely in `config.py`.
 2. `DataScraping.py` imports these credentials from `config.py` to initialize the Reddit API connection, keeping the sensitive information separate from the main logic.
 3. Now, if you need to update the credentials, you can do so in `config.py` without modifying the main script.
    

----------

## **Usage**

### **Run the Script**

 -  **DataScraping.py**
    
    
          1. This will scrape Reddit comments and save the data.
          
          2. Generates sentiment scores and prepares the data for modeling.
          
          3. Trains the machine learning model and evaluates its performance.
          
          4. Displays graphs of sentiment trends and model predictions.

    
## **Jupyter Notebooks**

In addition to the Python scripts, I have uploaded Jupyter notebooks that contain the same code. The notebooks provide an interactive environment for running the code step-by-step, and they also contain additional explanations and visualizations for better understanding. The notebooks are in the repository under the `notebook.ipynb` folder.



----------

## **Project Workflow**

### **1. Data Collection**

-   Used the `PRAW` library to scrape Reddit discussions from the `r/walstreetbets` subreddit.
-   Preprocessed data by removing special characters, links, and unnecessary whitespaces.

### **2. Sentiment Analysis**

-   Extracted sentiment polarity and subjectivity scores using `TextBlob` from the `NLTK` library.

### **3. Machine Learning**

-   Built a logistic regression model to classify sentiment data as predicting stock price movements.

### **4. Visualization**

-   Generated visual representations of:
    -   Sentiment polarity distribution.
    -   Correlation between sentiment and stock price 
movement.

----------

## **Results**

### **Model Performance**

|  |  |
|--|--|
| **Metric** |**Values**  |
**Accuracy**       |85%|
| **Precision**    |83%|
|**Recall**        |82%|
|** F1 Score**     |82%|



### **Visualization**

**1. Sentiment Distribution Visualization**

![image](https://github.com/user-attachments/assets/13354997-c5b2-4ade-971d-e8df9878f8fa)

**2. Feature Importance Visualization**


![image](https://github.com/user-attachments/assets/15202c75-c9e8-4546-af4c-03380e18db3d)


----------

## **Future Enhancements**

-   Expand data sources to include **Twitter** and **Telegram**.
-   Implement advanced models like **XGBoost** or **LSTMs** for better time-series forecasting.
-   Enhance feature extraction with **topic modeling** and **named entity recognition (NER)**.

----------

## **Contributing**

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch:
      
    `git checkout -b feature-branch` 
    
3.  Commit your changes:
    
    `git commit -m "Add your message here"` 
    
4.  Push to the branch:
    
    `git push origin feature-branch` 
    
5.  Create a pull request.

----------

### **Author**

Tushar Saxena



[![LinkedIn](https://github.com/user-attachments/assets/83d2f006-d33b-4994-9f1c-45757ba37462)](https://www.linkedin.com/in/tushar-saxena0410/)

----------
