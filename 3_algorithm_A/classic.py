import pandas as pd
import joblib
import inquirer
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score

# Scelta del modello
# Facciamo scelgiere all'utente quale modello far girare, oppure tutti
# poi continuiamo con l'esecuzione prescelta
models_options = [
    inquirer.List('model',
        message='Seleziona un modello da eseguire:',
        choices=[
            'Logistic Regression', 
            'Decision Tree', 
            'Random Forest', 
            'Support Vector Machines (SVM)', 
            'All (tutti in sequenza)'
        ]
    )
]
answers = inquirer.prompt(models_options)
chosen_option = answers['model']

# Load processed dataset
print('Loading dataset...')
dataset = pd.read_pickle("../DATA/processed_training.pkl.zip") 

# Tf-Idf Encoding
print('Tf-Idf Encoding...')
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
print('Tain/test train_test_split...')
X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, test_size=0.2, random_state=49)

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

# ############ MODEL #################

models_path = "../MODELS" # Path per salvare i modelli

# Funzioni per eseguire i modelli

def lr():
    print('Running LogisticRegression')
    lr_model = LogisticRegression()
    train(lr_model, 'LogisticRegression')
    confusion_matrix(lr_model, 'LogisticRegression')
    predict = lr_model.predict(X_test)
    model_report(lr_model, predict)
    show_roc_curve(lr_model, predict, 'LogisticRegression')
    lr_model_file = f'{models_path}/lr_model.sav'
    joblib.dump(lr_model, lr_model_file)

def dt():
    print('Running DecisionTreeClassifier')
    dectree_model = DecisionTreeClassifier(max_depth=58,random_state=42)
    train(dectree_model, 'DecisionTreeClassifier')
    confusion_matrix(dectree_model, 'DecisionTreeClassifier')
    predict = dectree_model.predict(X_test)
    model_report(dectree_model, predict)
    show_roc_curve(dectree_model, predict, 'DecisionTreeClassifier')
    dectree_model_file = f'{models_path}/dectree_model.sav'
    joblib.dump(dectree_model, dectree_model_file)

def rf():
    print('Running RandomForestClassifier')
    ranfor_class= RandomForestClassifier(random_state=42)
    
    params={
        "n_estimators": 3, # Più si aumenta, meglio è
        "max_depth": 58 # same as decision
    }

    ranfor_model = GridSearchCV(
        ranfor_class,
        param_grid= params,
        cv= 5,
        n_jobs= -1,
        verbose=1
    )    
    train(ranfor_model, 'RandomForestClassifier')
    confusion_matrix(ranfor_model, 'RandomForestClassifier')
    predict = ranfor_model.predict(X_test)
    model_report(ranfor_model, predict)
    show_roc_curve(ranfor_model, predict, 'RandomForestClassifier')
    ranfor_model_file = f'{models_path}/ranfor_model.sav'
    joblib.dump(ranfor_model, ranfor_model_file)

def svm():
    print('Running SupportVectorMachinesClassifier')
    svm_model = SVC(probability=True) # Without enabling probability, a ROC curve cannot be constructed
    train(svm_model, 'SupportVectorMachinesClassifier')
    confusion_matrix(svm_model, 'SupportVectorMachinesClassifier')
    predict = svm_model.predict(X_test)
    model_report(svm_model, predict)
    show_roc_curve(svm_model, predict, 'SupportVectorMachinesClassifier')
    svm_model_file = f'{models_path}/svm_model.sav'
    joblib.dump(svm_model, svm_model_file)

time_start = time.time() # Vediamo quanto tempo impiegano i modelli

if chosen_option == 'Logistic Regression':
    lr()
elif chosen_option == 'Decision Tree':
    dt()
elif chosen_option == 'Random Forest':
    rf()
elif chosen_option == 'Support Vector Machines (SVM)':
    svm()
elif chosen_option == 'All (tutti in sequenza)':
    lr()
    dt()
    rf()
    svm()

time_end = time.time()
exec_time = time_end - time_start

ore = int(exec_time // 3600)
minuti = int((exec_time % 3600) // 60)
secondi = int(exec_time % 60)
    
print(f"Tempo impiegato per l'esecuzione: {ore} ore, {minuti} minuti, {secondi} secondi")

'''
Sulla mia macchina hanno impiegato:

* Logistic Regression:              0 ore, 3 minuti, 27 secondi
* Decision Tree:                    0 ore, 8 minuti, 54 secondi
* Random Forest:                    
* Support Vector Machines (SVM):    
'''