import pandas as pd
import inquirer
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random


# Scelta del modello e di cosa analizzare
# Facciamo scelgiere all'utente quale modello far girare, se BERT o DilBERT
# e se analizzare solo i titoli o tutto il contenuto
# poi continuiamo con l'esecuzione prescelta

models_options = [
    inquirer.List('model',
        message='Seleziona un modello da eseguire:',
        choices=[
            'DistilBERT sui soli titoli', 
            'DistilBERT su titoli e corpo', 
            'BERT sui soli titoli', 
            'BERT su titoli e corpo'
        ]
    )
]
answers = inquirer.prompt(models_options)
chosen_option = answers['model']

use_content = False
distilled = True
    
if chosen_option == 'DistilBERT sui soli titoli':
    use_content = False
    distilled = True
elif chosen_option == 'DistilBERT su titoli e corpo':
    use_content = True
    distilled = True
elif chosen_option == 'BERT sui soli titoli':
    use_content = False
    distilled = False
elif chosen_option == 'BERT su titoli e corpo':
    use_content = True
    distilled = False

time_start = time.time() # Vediamo quanto tempo impiega il modello

# Load the dataset
dataset = pd.read_csv("../DATA/train.csv") # Carichiamo il dataset com'è, non quello processato

# Non ci devono essere campi spurii
dataset = dataset[dataset['text'].notna()]
dataset = dataset[dataset["author"].notna()]
dataset = dataset[dataset["title"].notna()]

# Funzione di help per il seeding
# Questo assicura la riporducibilità anche in presenza di riavvii

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # this function is safe to call even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

set_seed(42) # Si può cambiare a piacere

model_name = "distilbert-base-uncased" if distilled else "bert-base-uncased"

# Usiamo un max lenght per tagiare il testo, nelle prove può essere settato più basso, per tagliare le risorse.
# Un 512 sarebbe più opportuno che un 256 avendo le risorse necessarie.

max_length = 256 # max lenght for each document sample

# tokenizer load
if distilled:
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
else:
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

def prepare_data(dataset, use_content=False, test_size=0.2):
    texts = []
    labels = []
    for i in range(len(dataset)):
        text = dataset["title"].iloc[i]
        if use_content is True:
            text = dataset["author"].iloc[i] + " " + text
            text = dataset["text"].iloc[i] + " " + text
        label = dataset["label"].iloc[i]
        if text and label in [0, 1]: # controlliamo non ci siano errori
            texts.append(text)
            labels.append(label)
    return train_test_split(texts, labels, test_size=test_size)

train_texts, test_texts, train_labels, test_labels = prepare_data(dataset, use_content, test_size=0.2)

# Tokenizzazione
# Se è più di max_lenght tagliamo, se meno grande riempiamo con zeri.
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

# Trasformiamo gli encodings in una classe di torch Dataset
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

# Load model
if distilled:
    brt_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
else:
    brt_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Funzione per calclare le metriche
def get_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    } 

# Di seguito definiamo i parametri del training. 
# Questi possono essere settati in vari modi per una migliore resa sul dataset in oggetto

tr_args = TrainingArguments(
    output_dir='./bert_results',     # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=10,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    save_steps=200,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

# Prepariamo il trainer
trainer = Trainer(
    model=brt_model,                  # the instantiated Transformers model to be trained
    args=tr_args,                     # training arguments, defined above
    train_dataset=train_dataset,      # training dataset
    eval_dataset=test_dataset,        # testing dataset
    compute_metrics=get_metrics,      # the callback that computes metrics
)

# Fianlmente si esegue il training di BERT
print("\n----------")
print("START TRAINING")
print("----------\n")

trainer.train()

metrics = trainer.evaluate()
print(metrics)

# Salviamo il modello
models_path = "../MODELS" # Path per salvare i modelli
brt_model.save_pretrained(models_path)
tokenizer.save_pretrained(models_path)

# Info sul modello
import os
import io
file_path = "../MODELS/.model.txt"
text = "Distilled" if distilled else "Regular"
with open(file_path, "w") as file:
    file.write(text)

# Tempi di esecuzione
time_end = time.time()
exec_time = time_end - time_start

ore = int(exec_time // 3600)
minuti = int((exec_time % 3600) // 60)
secondi = int(exec_time % 60)
    
print(f"Tempo impiegato per l'esecuzione: {ore} ore, {minuti} minuti, {secondi} secondi")

'''
Sulla mia macchina hanno impiegato:

* DistilBERT sui soli titoli:    7 ore, 24 minuti, 22 secondi
* DistilBERT su titoli e corpo:  
* BERT sui soli titoli:          
* BERT su titoli e corpo:        
'''
