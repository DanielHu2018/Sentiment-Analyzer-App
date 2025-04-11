## Sentiment Analyzer App

A web-based Sentiment Analyzer built using Python (3.12.7). This application allows users to input text and receive a sentiment classification (Positive or Negative) based on natural language processing techniques.

## 🔧 Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/DanielHu2018/Sentiment-Analyzer-App.git
   cd Sentiment-Analyzer-App
   
2. **Download Datasets**: The dataset was too big to upload. Follow this link to download the dataset I used: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews.
  
3. Rename 'test.csv' to 'emotion_data_3_test.csv'.
4. Rename 'train.csv' to 'emotion_data_3.csv'.
5. After downloading, put the files in C:/Sentiment-Analyzer-App/data/raw.
  
6. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

7. **Run the App**:
   ```bash
   streamlit run app.py

## 📌 Todo / Future Improvements
Find dataset to train a model to classify text as informal or formal.
