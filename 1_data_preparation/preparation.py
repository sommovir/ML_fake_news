import re
import nltk
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
 
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load the dataset
dataset = pd.read_csv("../DATA/train.csv")
# _vecchio_dataset = pd.read_csv("../_vecchio_tutorial/news.csv")

print("Dataset info and stats")
print("------------")

print("Shape:", dataset.shape)
# print("vecchio dataset", _vecchio_dataset.shape)

# colonne
print("Columns names: ", dataset.columns)
# ['id', 'title', 'author', 'text', 'label']

# 'text' contiene gli articoli
print("\nStatistiche sulla lunghezza degli articoli:")
text_len = dataset.text.str.split().str.len()
print(text_len.describe())

# 'title' contiene i titoli degli articoli
print("\nStatistiche sulla lunghezza dei titoli:")
title_len = dataset.title.str.split().str.len()
print(title_len.describe())

# Distibuzione della classificazione
print("\nLabels distribution:")
print(dataset.label.value_counts())
print("1: means fake news (unreliable)")
print("0: means non-fake news (reliable)")

label_count_plot = sns.countplot(
    x= "label", 
    data=dataset,
    width=0.4
)
fig = label_count_plot.get_figure()
fig.legend(title='Classes', loc='upper right', labels=['reliable', 'unreliable'])
fig.savefig("img/label_count_plot.png") 


# Pulizia dei dati

# null data
print("Null data: ", dataset.isna().sum())
# replace with a space
dataset.fillna(" ", inplace= True)

# dataset['content'] = dataset['title'] + " " + dataset['author'] + " " + dataset['text']
dataset['content'] = dataset['title'] + " " + " " + dataset['text']

port_stem = PorterStemmer()

def preprocessing(content):
    content = re.sub('https?[\w:/\.]+',' ', content) # remove url
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

dataset['content'] = dataset.title.apply(preprocessing)

# Exlorative analysis

wordcloud = WordCloud(background_color='black', width=1200, height=800)

content_set = " ".join(dataset['content'])
cloud = wordcloud.generate(content_set)
plt.figure(figsize=(20,30))
plt.imshow(cloud)
plt.axis('off')
plt.savefig('img/content_word_cloud.png', bbox_inches='tight')

# Show only the reliable set
reliable_set = " ".join(dataset[dataset['label']==0]['content'])
reliable_cloud = wordcloud.generate(reliable_set)
plt.figure(figsize=(20,30))
plt.imshow(reliable_cloud)
plt.axis('off')
plt.savefig('img/content_word_cloud_reliable.png', bbox_inches='tight')

# Show only the unreliable set
unreliable_set = " ".join(dataset[dataset['label']==1]['content'])
unreliable_cloud = wordcloud.generate(unreliable_set)
plt.figure(figsize=(20,30))
plt.imshow(unreliable_cloud)
plt.axis('off')
plt.savefig('img/content_word_cloud_unreliable.png', bbox_inches='tight')

# N-grams analysis

def plot_ngrams(text, title, out, ylabel, xlabel="Occurences", n=2, top=20):
    count = (pd.Series(nltk.ngrams(text.split(), n)).value_counts())[:top] #get the top n ngrams
    count.sort_values().plot.barh(color='blue', width=.9, figsize=(20,30))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(out, bbox_inches='tight')
    #plt.show()
    
# Plot bigrams for the reliable set
plot_ngrams(reliable_set, "Top 20 reliable news bigrams", 'img/reliable_bigrams.png', 'Bigram', n=2)

# Plot bigrams for the unreliable set
plot_ngrams(unreliable_set, "Top 20 unreliable news bigrams", 'img/unreliable_bigrams.png', 'Bigram', n=2)

# Plot trigrams for the reliable set
plot_ngrams(reliable_set, "Top 20 reliable news trigrams", 'img/reliable_trigrams.png', 'Trigram', n=3)

# Plot trigrams for the unreliable set
plot_ngrams(unreliable_set, "Top 20 unreliable news trigrams", 'img/unreliable_trigrams.png', 'Trigram', n=3)