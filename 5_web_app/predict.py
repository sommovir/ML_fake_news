import pandas as pd
import time
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import io
import sys

file_path = "../MODELS/.model.txt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Init model...")
# These must be the same used in training the model
models_path = "../MODELS" # Path where the models are saved
max_length = 256 # max lenght for each document sample

with open(file_path, "r") as file:
    content = file.read()

if content == "Distilled":
    # load the tokenizer: DisltilBert
    tokenizer = DistilBertTokenizerFast.from_pretrained(models_path, do_lower_case=True)
    model = DistilBertForSequenceClassification.from_pretrained(models_path, num_labels=2)
elif content == "Regular":
    # load the tokenizer: Bert
    tokenizer = BertTokenizerFast.from_pretrained(models_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(models_path, num_labels=2)
else:
    print("Impossibile determinare il tipo di modello da caricare.")
    sys.exit()
    
model = model.to(device)

print("Model loaded successfully!")

def get_prediction(text, convert_to_label=False):
    time_start = time.time()
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    time_end = time.time()
    exec_time = time_end - time_start
    
    return d[int(probs.argmax())] if convert_to_label else int(probs.argmax()), exec_time

def get_clean_prediction(text, convert_to_label=False):
    r = get_prediction(text, convert_to_label)
    return r[0]

if __name__ == '__main__':
    print("Starting prediction on the test.csv ...")
    time_start = time.time() # Vediamo quanto tempo impiega a predire il test dataset

    # read the test set
    test_df = pd.read_csv("../DATA/test.csv")
    # make a copy of the testing set
    new_df = test_df.copy()
    # add a new column that contains the author, title and article content
    new_df["new_text"] = new_df["author"].astype(str) + " : " + new_df["title"].astype(str) + " - " + new_df["text"].astype(str)
    # get the prediction of all the test set
    new_df["label"] = new_df["new_text"].apply(get_clean_prediction)
    # make the submission file
    final_df = new_df[["id", "label"]]
    final_df.to_csv("../DATA/submit_final.csv", index=False)

    time_end = time.time()
    exec_time = time_end - time_start

    ore = int(exec_time // 3600)
    minuti = int((exec_time % 3600) // 60)
    secondi = int(exec_time % 60)
        
    print(f"Tempo impiegato per l'esecuzione: {ore} ore, {minuti} minuti, {secondi} secondi")
