import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score

# Load processed dataset
dataset = pd.read_pickle("../DATA/processed_training.pkl.zip") 

# Tf-Idf Encoding
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(dataset['content'].values)
tfidf = transformer.fit_transform(counts)

# Get targets
targets = dataset['label'].values

# Statistiche sul target e l'encoding Tf-Idf
print(f"target shape: {targets.shape}")
print(f"X shape: {tfidf.shape}")

# Tain/test train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, test_size=0.2, random_state=49)

# Statistiche sul train/test split
print(f"The shape of X_train is: {X_train.shape[0]}")
print(f"The shape of X_test is: {X_test.shape[0]}")

# ######### HELPER FUNCTIONS ##########

# train() esegue il training del modello
def train(model , model_name):
    model.fit(X_train,y_train)
    print(f"Training accuracy for\t{model_name} is\t{model.score(X_train,y_train)}")
    print(f"Testing accuracy for\t{model_name} is\t{model.score(X_test,y_test)}")

# confusion_matrix() mostra la matrice di confusione del modello
def confusion_matrix(model, model_name):
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
    disp.plot()
    plt.savefig(f'img/{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.clf()

# model_report() mostra i dati di accuratezza del modello
def model_report(model, predict):
    print(classification_report(
        y_test,
        predict
    ))

# show_roc_curve() mostra la ROC curve del modello
def show_roc_curve(model, y_pred, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Calcola la probabilità predetta per la classe positiva:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) # Calcola la curva ROC
    auc = roc_auc_score(y_test, y_pred_proba) # Calcola l'AUC (Area Under the Curve)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'img/{model_name}_roc_curve.png', bbox_inches='tight')
    plt.clf()
   
# LINEAR REGRESSION
lr_model = LogisticRegression()

train(lr_model, 'LogisticRegression')

confusion_matrix(lr_model, 'LogisticRegression')

lr_predict = lr_model.predict(X_test)
model_report(lr_model, lr_predict)

show_roc_curve(lr_model, lr_predict, 'LogisticRegression')

# Save the model

models_path = "../MODELS"
lr_model_file = f'{models_path}/lr_model.sav'
joblib.dump(lr_model, lr_model_file)

