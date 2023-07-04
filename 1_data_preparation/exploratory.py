import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import spacy
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

# Load spacy NLP model
# Shell: python -m spacy download en_core_web_sm 
#nlp = spacy.load('en_core_web_sm')

# Load processed dataset
dataset = pd.read_pickle("../DATA/processed_training.pkl.zip")

# ----------------------------------
# Distibuzione della classificazione
# ----------------------------------
# Utile per conoscere nel dataset quante osservazioni co sono per classe

print("Plotting Classification distribution...")

label_count_plot = sns.countplot(
    x= "label", 
    data=dataset,
    width=0.4
)
fig = label_count_plot.get_figure()
fig.legend(title='Classes', loc='upper right', labels=['reliable', 'unreliable'])
fig.savefig("img/label_count_plot.png")

# Exploratory analysis

# ---------------------
# 1. Analisi univariata
# ---------------------

# Esempi di analisi univariata:
# - Frequenza delle parole (generale e per categoria)
# - Lunghezza delle parole
# - Analisi delle frasi
# Mentre la frequenza delle parole potrebbe aiutare una valutazione relativa alla 
# classificazione in oggetto, l'analisi sulla lunghezza delle parole e l'analisi 
# fraseologica riguardano più un apporccio allo stile.
# Siccome lo stile appare ininfluente sulla classificazione delle fake news, 
# saltiamo l'analisi stilistica

# ---------------------------------------------------------------
# Analisi della frequenza delle parole (graficata con Word Cloud)
# ---------------------------------------------------------------

wordcloud = WordCloud(background_color='white', width=1200, height=800)

print("Plotting General content word cloud...")

content_set = " ".join(dataset['content'])
cloud = wordcloud.generate(content_set)
plt.figure(figsize=(20,30))
plt.imshow(cloud)
plt.axis('off')
plt.savefig('img/general_content_word_cloud.png', bbox_inches='tight')
plt.clf()

# Show only the reliable set
print("Plotting content word cloud for reliable set...")

reliable_set = " ".join(dataset[dataset['label']==0]['content'])
reliable_cloud = wordcloud.generate(reliable_set)
plt.figure(figsize=(20,30))
plt.imshow(reliable_cloud)
plt.axis('off')
plt.savefig('img/content_word_cloud_reliable.png', bbox_inches='tight')
plt.clf()

# Show only the unreliable set
print("Plotting content word cloud for unreliable set...")

unreliable_set = " ".join(dataset[dataset['label']==1]['content'])
unreliable_cloud = wordcloud.generate(unreliable_set)
plt.figure(figsize=(20,30))
plt.imshow(unreliable_cloud)
plt.axis('off')
plt.savefig('img/content_word_cloud_unreliable.png', bbox_inches='tight')
plt.clf()

# -----------------------
# 2. Analisi multivariata
# -----------------------

# Esempi di analisi multivariata
# - Analisi delle co-occorrenze assolute e relative (alla categoria da classificare)
# - Analisi dei topic (argomenti)
# - Sentiment analysis (analisi del sentimento)
# - Analisi delle associazioni
# Esporeremo in dettaglio tutte queste analisi

# ------------------------------------------------------------------
# N-grams analysis (Analisi delle co-occorrenze assolute e relative)
# ------------------------------------------------------------------

def plot_ngrams(text, title, out, ylabel, xlabel="Occurences", n=2, top=20):
    count = (pd.Series(nltk.ngrams(text.split(), n)).value_counts())[:top] #get the top n ngrams
    count.sort_values().plot.barh(
        color='blue', 
        width=.9, 
        figsize=(12,8),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title
    )
    plt.savefig(out, bbox_inches='tight')
    #plt.show()
    plt.clf()
    
# Plot bigrams for the reliable set
print("Plotting bigrams for reliable set...")
plot_ngrams(reliable_set, "Top 20 reliable news bigrams", 'img/reliable_bigrams.png', 'Bigram', n=2)

# Plot bigrams for the unreliable set
print("Plotting bigrams for ureliable set...")
plot_ngrams(unreliable_set, "Top 20 unreliable news bigrams", 'img/unreliable_bigrams.png', 'Bigram', n=2)

# Plot trigrams for the reliable set
print("Plotting trigrams for reliable set...")
plot_ngrams(reliable_set, "Top 20 reliable news trigrams", 'img/reliable_trigrams.png', 'Trigram', n=3)

# Plot trigrams for the unreliable set
print("Plotting trigrams for unreliable set...")
plot_ngrams(unreliable_set, "Top 20 unreliable news trigrams", 'img/unreliable_trigrams.png', 'Trigram', n=3)

# -----------------
# Analisi dei topic 
# -----------------

def topic_analysis(text):
    doc = nlp(text)
    return [
        token.text
        for token in doc
        if token.is_stop == False
        and token.is_punct == False
        and token.is_space == False
        and token.pos_ == 'NOUN'
    ]

dataset['topics'] = dataset['content'].apply(topic_analysis)

# Grafichiamo i risultati
wordcloud = WordCloud(background_color='white', width=1200, height=800)

# Unisce le liste dei topics in una stringa per riga
def joins_txt_list(txt_list):
    return ' '.join(txt_list)
dataset['topics'] = dataset['topics'].apply(joins_txt_list)

reliable_set_topics = " ".join(dataset[dataset['label']==0]['topics'])
reliable_topic_cloud = wordcloud.generate(reliable_set_topics)
plt.figure(figsize=(20,30))
plt.imshow(reliable_topic_cloud)
plt.axis('off')
plt.savefig('img/topics_word_cloud_reliable.png', bbox_inches='tight')
plt.clf()

unreliable_set_topics = " ".join(dataset[dataset['label']==1]['topics'])
unreliable_topic_cloud = wordcloud.generate(unreliable_set_topics)
plt.figure(figsize=(20,30))
plt.imshow(unreliable_topic_cloud)
plt.axis('off')
plt.savefig('img/topics_word_cloud_unreliable.png', bbox_inches='tight')
plt.clf()

# Eliminiamo i topic dati da parole comuni
# Lista di parole comuni da rimuovere
common_words = ["time", "year", "day", "thing"]  

wordcloud_stopwords = WordCloud(stopwords=set(common_words),background_color='white', width=1200, height=800)

# Topics per il reliable set
reliable_topic_cloud = wordcloud_stopwords.generate(reliable_set_topics)
plt.figure(figsize=(20,30))
plt.imshow(reliable_topic_cloud)
plt.axis('off')
plt.savefig('img/topics_word_cloud_reliable_post.png', bbox_inches='tight')
plt.clf()

# Topics per il unreliable set
unreliable_topic_cloud = wordcloud_stopwords.generate(unreliable_set_topics)
plt.figure(figsize=(20,30))
plt.imshow(unreliable_topic_cloud)
plt.axis('off')
plt.savefig('img/topics_word_cloud_unreliable_post.png', bbox_inches='tight')
plt.clf()

# ------------------
# Sentiment analysis
# ------------------

# Download Vadel Lexicon
nltk.download('vader_lexicon')
# Crea un'istanza del SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def sentiment_eval(testo):
    sentiment = sia.polarity_scores(testo)
    return sentiment['compound']
dataset['sentiment'] = dataset['content'].apply(sentiment_eval)

# Calcola la media del sentimento per ciascuna categoria
mean_sentiment_per_category = dataset.groupby('label')['sentiment'].mean()

fig, ax = plt.subplots()
mean_sentiment_per_category.plot(kind='bar', ax=ax)

# Centrato sullo zero
#y_min, y_max = ax.get_ylim()
#abs_max = max(abs(y_min), abs(y_max))
#ax.set_ylim(-abs_max, abs_max)

ax.set_xlabel('Categoria')
ax.set_ylabel('Sentimento Medio')
ax.set_xticklabels(['reliable', 'unreliable'], rotation=0)
plt.title('Distribuzione del Sentimento per Categoria')
plt.savefig('img/mean_sentiment_analysis_per_category.png', bbox_inches='tight')
plt.clf()

# --------------------------
# Analisi delle associazioni
# --------------------------
# Occhio prende davvero varie ore di elabrazione

# Visto che sulla colonna content impiega veramente tanto tempo, 
# è possibile estrarre associazioni dalla sola colonna dei titles

te = TransactionEncoder() # TransactionEncoder object

# Scegliere se analizziamo i soli titoli o il content = titoli + corpo dell'articolo
'''
# Quanto segue serve solo per analizzare i soli titoli
port_stem = PorterStemmer()
nltk.download('stopwords')
stop_words = stopwords.words('english')
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
dataset['title_prep'] = dataset.title.apply(preprocessing)
'''
# se usiamo il content il preprocessing è già stato fatto

# Analisi reliable
df_class_0 = dataset[dataset['label'] == 0]

# Scegliere se analizziamo i soli titoli o il content = titoli + corpo dell' articolo
#te_ary_0 = te.fit_transform(df_class_0['title_prep'].apply(lambda x: x.split())) # Solo titoli
te_ary_0 = te.fit_transform(df_class_0['content'].apply(lambda x: x.split())) # Titoli e corpo

df_transformed_0 = pd.DataFrame(te_ary_0, columns=te.columns_)
frequent_itemsets_class_0 = apriori(df_transformed_0, min_support=0.1, use_colnames=True)
rules_class_0 = association_rules(frequent_itemsets_class_0, metric="confidence", min_threshold=0.5)

# Analisi unreliable
df_class_1 = dataset[dataset['label'] == 1]

# Scegliere se analizziamo i soli titoli o il content = titoli + corpo dell' articolo
#te_ary_1 = te.fit_transform(df_class_1['title_prep'].apply(lambda x: x.split())) # Solo titoli
te_ary_1 = te.fit_transform(df_class_1['content'].apply(lambda x: x.split())) # Titoli e corpo

df_transformed_1 = pd.DataFrame(te_ary_1, columns=te.columns_)
frequent_itemsets_class_1 = apriori(df_transformed_1, min_support=0.1, use_colnames=True)
rules_class_1 = association_rules(frequent_itemsets_class_1, metric="confidence", min_threshold=0.5)

# Grafico degli itemset frequenti nelle reliable news
plt.figure(figsize=(10, 8))
plt.barh(range(len(frequent_itemsets_class_0)), frequent_itemsets_class_0['support'], align='center')
plt.yticks(range(len(frequent_itemsets_class_0)), frequent_itemsets_class_0['itemsets'].apply(lambda x: ', '.join(x)))
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.title('Frequent Itemsets (reliable news)')
plt.savefig('img/associations_frequent_itemset_reliable.png', bbox_inches='tight')
plt.clf()

# Grafico degli itemset frequenti nelle unreliable news
plt.figure(figsize=(10, 1))
plt.barh(range(len(frequent_itemsets_class_1)), frequent_itemsets_class_1['support'], align='center')
plt.yticks(range(len(frequent_itemsets_class_1)), frequent_itemsets_class_1['itemsets'].apply(lambda x: ', '.join(x)))
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.title('Frequent Itemsets (unreliable news)')
plt.savefig('img/associations_frequent_itemset_unreliable.png', bbox_inches='tight')
plt.clf()
