import pandas as pd
import numpy as np
import inquirer
import joblib
import time
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Scelta del modello
# Facciamo scelgiere all'utente quale modello far girare
# poi continuiamo con l'esecuzione prescelta
models_options = [
    inquirer.List('model',
        message='Seleziona un modello da eseguire:',
        choices=[
            'Logistic Regression', 
            'Decision Tree', 
            'Random Forest', 
            'Support Vector Machines (SVM)'
        ]
    )
]
answers = inquirer.prompt(models_options)
chosen_option = answers['model']

# Timing
time_start = time.time()

nltk.download('stopwords')
stop_words = stopwords.words('english')

# It must be the same used in training the model
models_path = "../MODELS/Classic_predict" # Path where the models are saved

# Load the general dataset
dataset = pd.read_csv("../DATA/train.csv")

# read the test dataset
test_df  = pd.read_csv("../DATA/test.csv")

# Pulizia dei dati
print('Pre-processing...')

dataset.fillna(" ", inplace= True)
test_df.fillna(" ", inplace= True)

# add a new column that contains the author, title and article content
dataset['content'] = dataset['title'] + " " + dataset['author'] + " " + dataset['text']
test_df["new_text"] = test_df["author"].astype(str) + " : " + test_df["title"].astype(str) + " - " + test_df["text"].astype(str)


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
test_df["new_text"] = test_df.new_text.apply(preprocessing)

# Apply TF-IdF
print('Tf-Idf Encoding...')

transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
# Combine training and prediction data
combined_data = np.concatenate([dataset['content'].values, test_df['new_text'].values], axis=0)

# Fit the CountVectorizer on the combined training and prediction data
count_vectorizer.fit(combined_data)
# Fit the TfidfTransformer on the combined training and prediction data
transformer.fit(count_vectorizer.transform(combined_data))

# Apply the transformers to training and prediction data separately
training_features = transformer.transform(count_vectorizer.transform(dataset['content'].values))
prediction_features = transformer.transform(count_vectorizer.transform(test_df['new_text'].values))

# Save Tf-Idf transformer and vectorizer
joblib.dump(count_vectorizer, f'{models_path}/count_vectorizer.sav')
joblib.dump(transformer, f'{models_path}/transformer.sav')

# Get targets
targets = dataset['label'].values

# Tain/test train_test_split
print('Tain/test train_test_split...')
X_train, X_test, y_train, y_test = train_test_split(training_features, targets, test_size=0.2, random_state=49)

# train() esegue il training del modello
def train(model , model_name):
    model.fit(X_train,y_train)


# Funzioni per eseguire i modelli

def lr():
    print('Running LogisticRegression')
    lr_model = LogisticRegression()
    train(lr_model, 'LogisticRegression')
    lr_model_file = f'{models_path}/lr_model.sav'
    joblib.dump(lr_model, lr_model_file)
    return lr_model

def dt():
    print('Running DecisionTreeClassifier')
    dectree_model = DecisionTreeClassifier(max_depth=58,random_state=42)
    train(dectree_model, 'DecisionTreeClassifier')
    dectree_model_file = f'{models_path}/dectree_model.sav'
    joblib.dump(dectree_model, dectree_model_file)
    return dectree_model

def rf():
    print('Running RandomForestClassifier')
    ranfor_class= RandomForestClassifier(random_state=42)
    
    params={
        "n_estimators": range(3), # Più si aumenta, meglio è
        "max_depth": range(58) # same as decision
    }

    ranfor_model = GridSearchCV(
        ranfor_class,
        param_grid= params,
        cv= 5,
        n_jobs= -1,
        verbose=1
    )    
    train(ranfor_model, 'RandomForestClassifier')
    ranfor_model_file = f'{models_path}/ranfor_model.sav'
    joblib.dump(ranfor_model, ranfor_model_file)
    return ranfor_model

def svm():
    print('Running SupportVectorMachinesClassifier')
    svm_model = SVC(probability=True) # Without enabling probability, a ROC curve cannot be constructed
    train(svm_model, 'SupportVectorMachinesClassifier')
    svm_model_file = f'{models_path}/svm_model.sav'
    joblib.dump(svm_model, svm_model_file)
    return svm_model


def predict(model):    
    print("Starting prediction on the test.csv ...")
    
    # get the prediction of all the test set
    print('Predicting...')
    predictions = model.predict(prediction_features)
    test_df['label'] = predictions
    
    # make the submission file
    final_df = test_df[["id", "label"]]
    final_df.to_csv("../DATA/Classic_predict/submit_final.csv", index=False)


if chosen_option == 'Logistic Regression':
    model = lr()
    predict(model) 
elif chosen_option == 'Decision Tree':
    model = dt()
    predict(model)
elif chosen_option == 'Random Forest':
    model = rf()
    predict(model)
elif chosen_option == 'Support Vector Machines (SVM)':
    model = svm()
    predict(model)
else:
    print("Error: no model chosen")
    sys.exit(1)

time_end = time.time()
exec_time = time_end - time_start

ore = int(exec_time // 3600)
minuti = int((exec_time % 3600) // 60)
secondi = int(exec_time % 60)
    
print(f"Tempo impiegato per l'esecuzione: {ore} ore, {minuti} minuti, {secondi} secondi")

'''
Sulla mia macchina hanno impiegato:

* Logistic Regression:              
* Decision Tree:                    
* Random Forest:                    
* Support Vector Machines (SVM):    
'''