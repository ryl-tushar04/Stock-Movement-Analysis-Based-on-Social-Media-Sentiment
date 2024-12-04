import praw
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import config

# Initialize Reddit API
reddit = praw.Reddit(
client_id=config.REDDIT_CLIENT_ID,
    client_secret=config.REDDIT_CLIENT_SECRET,
    user_agent=config.REDDIT_USER_AGENT,      )

# Scraping function
def scrape_reddit(subreddit_name, keyword, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.search(keyword, limit=limit):
        posts.append({
            "title": post.title,
            "selftext": post.selftext,
            "created_utc": post.created_utc,
            "score": post.score,
            "comments": post.num_comments
        })
    return pd.DataFrame(posts)

# Scrape data from Reddit
df = scrape_reddit("wallstreetbets", "stocks", limit=500)  # Example: Scrape 'r/wallstreetbets' for keyword "stocks"
df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')

# Save raw data for reference
df.to_csv("reddit_data.csv", index=False)

# Preprocessing and Sentiment Analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = " ".join([word for word in text if word.isalnum()])
        return text
    return ""

df['cleaned_text'] = df['title'] + " " + df['selftext']
df['cleaned_text'] = df['cleaned_text'].apply(preprocess_text)

df['sentiment'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment_category'] = df['sentiment'].apply(
    lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
)

# Save preprocessed data
df.to_csv("processed_reddit_data.csv", index=False)

# Feature Engineering for Prediction
df['sentiment_label'] = df['sentiment_category'].map({'negative': 0, 'neutral': 1, 'positive': 2})
X = df[['sentiment', 'score', 'comments']]
y = df['sentiment_label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

# Visualization: Sentiment Distribution
sns.countplot(df['sentiment_category'])
plt.title("Sentiment Distribution")
plt.show()

# Visualization: Feature Importance
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()
