# 📰 Fake News Detection using Naïve Bayes  

![Fake News Detector](https://media.giphy.com/media/xT5LMP3l9pSgUkWogw/giphy.gif)  

## 📌 Project Overview  
This project builds a **Fake News Detector** using **Natural Language Processing (NLP)** and **Machine Learning**. We use the **Naïve Bayes Classifier** trained on the `Fake.csv` dataset to classify news articles as **FAKE** or **REAL**.  

---

## 🚀 Installation  

Ensure you have Python installed, then install the required dependencies:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📂 Project Structure  
```
📁 fake-news-detection
│── 📄 fake_news_detection.py   # Main script
│── 📄 README.md                # Project documentation
│── 📊 Fake.csv                  # Dataset (should be placed here)
```

---

## 🛠️ Steps Involved  

### 1️⃣ Import Necessary Libraries  
```python
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

### 2️⃣ Load the Dataset  
```python
df = pd.read_csv("Fake.csv", on_bad_lines="skip", encoding='utf-8')

print(df.columns)  # Display column names
print(df['subject'].unique())  # Check unique values in 'subject'
df.rename(columns={'subject': 'label'}, inplace=True)  # Rename 'subject' to 'label'
```

---

### 🧹 3️⃣ Data Cleaning  
```python
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(r"https?://\S+|www\.\S+", '', text)  # Remove URLs
    text = re.sub(r"<.*?>+", '', text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", '', text)  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    return text

df['content'] = df['content'].apply(clean_text)
```

---

### ✂️ 4️⃣ Splitting the Data  
```python
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 🔡 5️⃣ Convert Text to Numerical Features  
```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

---

### 🔥 6️⃣ Train the Naïve Bayes Model  
```python
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

---

### 🔍 7️⃣ Making Predictions  
```python
y_pred = model.predict(X_test_tfidf)
```

---

### 📈 8️⃣ Model Evaluation  
```python
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

📌 **Sample Confusion Matrix:**  
![Confusion Matrix](https://i.imgur.com/ZwTHyFt.png)

---

### 📰 9️⃣ Fake News Detection Function  
```python
def predict_news(news_text):
    news_text = clean_text(news_text)
    news_vectorized = vectorizer.transform([news_text])
    prediction = model.predict(news_vectorized)
    return "FAKE NEWS" if prediction[0] == 1 else "REAL NEWS"

# Example Usage
news = "Breaking news! The direct dbt of 72000 per month will be given to each person"
print("\nPrediction:", predict_news(news))
```

---

## 🎯 Key Findings  
✅ **Data preprocessing (cleaning text) improved model accuracy.**  
✅ **TF-IDF vectorization helped in feature extraction.**  
✅ **The Naïve Bayes model effectively classifies fake news.**  

---

## 📌 Future Improvements  
🔹 Try different **machine learning models** (e.g., Random Forest, LSTM).  
🔹 Improve **text preprocessing** by removing stopwords.  
🔹 Deploy the model using Flask or Streamlit for a web-based interface.  

---

## 💡 Contributors  
👤 **Your Name** | [GitHub Profile](https://github.com/yourprofile)  

⭐ If you like this project, don't forget to give it a **star** on GitHub! ⭐  

---

🚀 **Happy Coding & Stay Informed!** 📰
