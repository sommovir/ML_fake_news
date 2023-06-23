import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
 
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load the dataset
dataset = pd.read_csv("../DATA/train.csv")

print("\nDataset info and stats")
print("------------")

print("Shape:", dataset.shape)

# colonne
print("Columns names: ", dataset.columns)
# ['id', 'title', 'author', 'text', 'label']

# 'label' contiene le classi: 0 non-fake, 1 fake
print("\nLabels distribution:")
print(dataset.label.value_counts())
print("1: means fake news (unreliable)")
print("0: means non-fake news (reliable)")

# 'text' contiene gli articoli
print("\nStatistiche sulla lunghezza degli articoli:")
text_len = dataset.text.str.split().str.len()
print(text_len.describe())

# 'title' contiene i titoli degli articoli
print("\nStatistiche sulla lunghezza dei titoli:")
title_len = dataset.title.str.split().str.len()
print(title_len.describe())

# Pulizia dei dati

# null data
print("Null data: ", dataset.isna().sum())
# replace with a space
dataset.fillna(" ", inplace= True)

# dataset['content'] = dataset['title'] + " " + dataset['author'] + " " + dataset['text']
dataset['content'] = dataset['title'] + " " + " " + dataset['text']

print("\nData cleaning...")

port_stem = PorterStemmer()

def preprocessing(content):
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ', content) # remove url
    content = re.sub('[^\.\w\s]',' ', content)       # remove all that is not a character or punctuation
    content = re.sub('[^a-zA-Z]',' ', content)       # replace all non-alphabetic with a space
    content = re.sub('\s\s+',' ', content)           # replace more than one space with a single space
    content = content.lower()                        # Convert to lower case
    
    # Split the words into list
    content_list = content.split()
    
    #generate a list of stemmed words, excluding stop words
    stemmed_content = [
        port_stem.stem(word)
        for word in content_list
        if word not in stop_words
    ]
    # Join elements in a single string space-separated
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

dataset['content'] = dataset.content.apply(preprocessing)

# Save processed dataset
dataset.to_pickle("../DATA/processed_training.pkl.zip")

