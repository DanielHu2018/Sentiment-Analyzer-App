import streamlit as st
import pandas as pd

### Title/Header ###
st.title("Sentiment Analyzer by Daniel Hu")
st.header("Codebase", divider="red")

### Sidebar ###
home = st.sidebar.page_link("app.py", label="Home", icon = "🏠")
results = st.sidebar.page_link("pages/results.py", label="Results", icon = "🚀")
code = st.sidebar.page_link("pages/codebase.py", label = "Codebase", icon = "🤖")

### Tabs in codebase ###
model, visualizations, data = st.tabs(["Model", "Visualization", "Data"])

### Data page ###
with data:
    data_path = "C:/SentimentAnalyzerApp/data/raw/emotion_data_3_test.csv"
    column_names = ['Sentiment (1 = Negative, 2 = Postive)', 'Review Title', 'Review Content']

    df = pd.read_csv(data_path, names = column_names) # Load data
    
    data.write("100 data entries from the test dataset")
    data.write(df.head(100))  # Display the first 100 rows of the CSV 

### Model page ###
model.write("Code for training the model")

# Code for training model
model_code = '''# Daniel Hu
# Categorizing sentiment in text using amazon reviews

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from joblib import dump
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
import re

# Load dataframes for training and testing
train_df = pd.read_csv("C:/SentimentAnalyzerApp/data/raw/emotion_data_3.csv", names = ['polarity','review_title','review_body']).dropna()
test_df =  pd.read_csv("C:/SentimentAnalyzerApp/data/raw/emotion_data_3_test.csv", names = ['polarity','review_title','review_body']).dropna()

# Keep sentiment and text, drop the rest.
train_df.drop(['review_title'], axis = 1)
test_df.drop(['review_title'], axis = 1)

# Set axes
x_train = train_df.review_body
y_train = train_df.polarity

x_test = test_df.review_body
y_test = test_df.polarity

# Replace numerical polarity values with 'positive' or 'negative'
y_train = y_train.replace({1:'negative', 2:'positive'})
y_test = y_test.replace({1:'negative', 2:'positive'})

# Pipeline with punctuation and capitalization
# Bag of words appproach with n-gram vocabulary
pipe = Pipeline([
  ('vec', CountVectorizer(stop_words='english', min_df=1000, analyzer = 'word', ngram_range=(1,2))),
  ('tfid', TfidfTransformer()),
  ('norm', Normalizer()),
  ('lr', SGDClassifier(loss='log_loss'))
])

# Train
model = pipe.fit(x_train, y_train)

dump(model, 'C:/SentimentAnalyzerApp/models/model1.joblib')
'''
model.code(model_code)

### Visualization Page ###
visualizations.write("Code for graphing the model's accuracy")


dfCode = '''# Daniel Hu
# Categorizing sentiment in text using amazon reviews

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import seaborn as sns


# Load dataframes for training and testing
test_df =  pd.read_csv("../data/raw/emotion_data_3_test.csv", names = ['polarity','review_title','review_body']).dropna()

# Keep sentiment and text, drop the rest.
test_df.drop(['review_title'], axis = 1)

# Set axes
x_test = test_df.review_body
y_test = test_df.polarity

# Load model
model = joblib.load("../data/raw/model1.joblib")

# Get sentiment predictions
y_pred = model.predict(x_test)

# Replace numerical polarity values with 'positive' or 'negative'
y_test = y_test.replace({1:'negative', 2:'positive'})'''

numericCF = '''# Compute numeric confusion matrix
num_cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])

# Display confusion matrix with labels (numeric)
num_disp = ConfusionMatrixDisplay(confusion_matrix=num_cm, display_labels=['[—]', '[+]'])

num_disp.plot(cmap = sns.diverging_palette(8, 253, s = 100, l = 54, as_cmap=True), values_format = "d")

# Highlight false positives and false negatives
plt.title("Numeric Confusion Matrix Diagram of Predicted vs. Expected Sentiment")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Expected Sentiment")
plt.show()'''


normCF = '''# Compute normalized confusion matrix
normal_cm = confusion_matrix(y_test, y_pred, normalize="true", labels=['negative', 'positive'])

# Display confusion matrix with labels (normalized)
norm_disp = ConfusionMatrixDisplay(confusion_matrix=normal_cm, display_labels=['[—]', '[+]'])
norm_disp.plot(cmap = sns.diverging_palette(8, 253, s = 100, l = 54, as_cmap=True), values_format = ".2f")

# Highlight false positives and false negatives
plt.title("Normalized Confusion Matrix Diagram of Predicted vs. Expected Sentiment")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Expected Sentiment")

plt.show()
'''

numericBar = '''# Extract values
TN_num, FP_num, FN_num, TP_num = num_cm.ravel()  # Unpack values

# Create labels and values
categories = ['Negative Reviews', 'Positive Reviews']
true_values_num = [TN_num, TP_num]  # True classifications
false_values_num = [FP_num, FN_num]  # Misclassifications (FP, FN)

# Bar chart
fig, ax = plt.subplots(figsize=(8, 10))
bar_width = 0.5

bars1_num = np.array(true_values_num)  # True predictions
bars2_num = np.array(false_values_num)  # False predictions

# Plot stacked bars
p1 = plt.bar(categories, bars1_num, color=['dodgerblue'], label="Correctly Classified", width=bar_width)
p2 = plt.bar(categories, bars2_num, bottom=bars1_num, color=['crimson'], label="Misclassified", width=bar_width)

# Labels & legend
plt.ylabel("Count")
plt.title("Numeric Stacked Bar Graph of Sentiment Classification")
plt.legend()

for r1, r2 in zip(p1, p2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")

plt.show()'''

normBar = '''# Extract values
TN_norm, FP_norm, FN_norm, TP_norm = normal_cm.ravel()  # Unpack values

# Create labels and values
categories = ['Negative Reviews', 'Positive Reviews']
true_values_norm = [TN_norm, TP_norm]  # True classifications
false_values_norm = [FP_norm, FN_norm]  # Misclassifications (FP, FN)

# Bar chart
fig, ax = plt.subplots(figsize=(8, 10))
bar_width = 0.5

bars1_norm = np.array(true_values_norm)  # True predictions
bars2_norm = np.array(false_values_norm)  # False predictions

# Plot stacked bars
p1 = plt.bar(categories, bars1_norm, color=['dodgerblue'], label="Correctly Classified", width=bar_width)
p2 = plt.bar(categories, bars2_norm, bottom=bars1_norm, color=['crimson'], label="Misclassified", width=bar_width)

# Labels & legend
plt.ylabel("Distribution")
plt.title("Normalized Stacked Bar Graph of Sentiment Classification")
plt.legend()

for r1, r2 in zip(p1, p2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%.2f" % h1, ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%.2f" % h2, ha="center", va="bottom", color="white", fontsize=12, fontweight="bold")

plt.show()'''
visualizations.code(dfCode)
cf1, cf2, bar1, bar2 = visualizations.tabs(["Numeric Confusion Matrix", "Normalized Confusion Matrix", "Numeric Bar Graph", "Normalized Bar Graph"])

cf1.code(numericCF)
cf2.code(normCF)
bar1.code(numericBar)
bar2.code(normBar)