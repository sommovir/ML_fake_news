import pandas as pd
import re
import string
import numpy

dataframe = pd.read_csv('news.csv')
print(dataframe.head()) #dataframe

x = dataframe['text']
y = dataframe['label']

print(x)
print(y)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn
from sklearn.preprocessing import Binarizer

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

dataframe['text'] = dataframe['text'].apply(word_drop)

print(dataframe['text'].head(10))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(y_train)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)

print(tfid_x_train)

classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train, y_train)
print("Training set score: {:.3f}".format(classifier.score(tfid_x_train, y_train)))
print("Test set score: {:.3f}".format(classifier.score(tfid_x_test, y_test)))

X_train_prediction = classifier.predict(tfid_x_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print(training_data_accuracy)

y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

print(classification_report(y_train, X_train_prediction))
print(classification_report(y_test, y_pred))

cf = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(cf)

nb = MultinomialNB()
nb.fit(tfid_x_train, y_train)
print("Training set score: {:.3f}".format(nb.score(tfid_x_train, y_train)))
print("Test set score: {:.3f}".format(nb.score(tfid_x_test, y_test)))

y_pred1 = nb.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred1)
print(f'Accuracy: {round(score*100,2)}%')

cf1 = confusion_matrix(y_test, y_pred1, labels=['FAKE', 'REAL'])
print(cf1)

rf = RandomForestClassifier()
rf.fit(tfid_x_train, y_train)
print("Training set score: {:.3f}".format(rf.score(tfid_x_train, y_train)))
print("Test set score: {:.3f}".format(rf.score(tfid_x_test, y_test)))

y_pred2 = rf.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred2)
print(f'Accuracy: {round(score*100,2)}%')

cf2 = confusion_matrix(y_test, y_pred2, labels=['FAKE', 'REAL'])
print(cf2)

svm = svm.SVC(kernel='linear', gamma='auto', C=2)
svm.fit(tfid_x_train, y_train)
print("Training set score: {:.3f}".format(svm.score(tfid_x_train, y_train)))
print("Test set score: {:.3f}".format(svm.score(tfid_x_test, y_test)))

y_pred3 = svm.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred3)
print(f'Accuracy: {round(score*100,2)}%')

cf3 = confusion_matrix(y_test, y_pred3, labels=['FAKE', 'REAL'])
print(cf3)

bnb = BernoulliNB(binarize=0.0)
bnb.fit(tfid_x_train, y_train)
print("Training set score: {:.3f}".format(bnb.score(tfid_x_train, y_train)))
print("Test set score: {:.3f}".format(bnb.score(tfid_x_test, y_test)))

y_pred4 = svm.predict(tfid_x_test)
score = accuracy_score(y_test, y_pred4)
print(f'Accuracy: {round(score*100,2)}%')

cf4 = confusion_matrix(y_test, y_pred4, labels=['FAKE', 'REAL'])
print(cf4)

r_probs = [0 for _ in range(len(y_test))]
logistic_probs = classifier.predict(tfid_x_test)

logistic_probs = logistic_probs[:, 1]

logistic_auc = roc_auc_score(y_test, logistic_probs)

print('Logistic Regression : AUROC = % 0.3f ' % (logistic_auc))

nb_probs = nb.predict_proba(tfid_x_test)

nb_probs = nb_probs[:, 1]

nb_auc = roc_auc_score(y_test, nb_probs)

print('Multinomial : AUROC = % 0.3f ' % (nb_auc))

fpr, tpr, thresholds = roc_curve(y_test, logistic_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)

fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')
fake_news_det("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)

import pickle
pickle.dump(classifier, open('model.pkl', 'wb'))

loaded_model = pickle.load(open('model.pkl', 'rb'))

def fake_news_det1(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)

fake_news_det1("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)
fake_news_det1("""U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.""")
