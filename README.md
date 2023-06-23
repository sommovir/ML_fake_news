# Fake News Detector

Valutazione algoritmi e creazione di una web app per la classificazione di articoli come fake news.

## Preparazione

Usiamo l'ambiente `pipenv` per la gestione del Python Environment.

```shell
pip install pipenv
```

Comando di attivazione:

```shell
pipenv shell
```

Comando di chiusura:

```shell
exit # o CTRL+D o quit, a seconda della shell
```

Comando per installare un pacchetto python nell'environment:

```shell
pipenv install <package-name>
```

## Download del Dataset

Si raccomanda di utilizzare lo strumento di Kaggle da riga di comando.

Comando per l'installazione (se non già disponibile sul sitema)

```shell
pip install --user kaggle
```

Comando per scaricare il dataset:

```shell
mkdir DATA && cd DATA
kaggle competitions download -c fake-news
```

Oppure scaricare dalla pagina della competizione direttamente nella cartella `DATA/`.
La cartella è inserita nel `.gitignore` per evitare che il dataset vada a intasare il repository.

## 1 Data Preparation

In questa sezione si svolge l'analisi preliminare dei dati, la pulizia e normalizzazione degli stessi.

Svolgiamo anche alcune funzioni di Exlorative Data Analysis, in particolar modo:

- **Univariate Analysis**: useremo un grafico Word LCoud per mostrare la frequesnza di certe parole all'interno del testo da classificare
- **Bivariate Analysis**: faremo un'analisi dei bigrammi più frequenti

## 2 Baseline

In questa sezione stabiliamo la baseline, usualmenteun linear o logistic regression model.
Eseguiamo tutti i benchmark soliti per i modelli, così abbiamo dati per inferire la validità di modelli susseguenti a partire dal modello prescelto


## 3 Algorithm_A

Qui valutiamo vari algoritmi classici per stabilire quale sia uno buono che possa competere nella sezione classici

Diamo la possibilità all'utente di scegliere quale eseguire perchè prende tempo e qualcuno potrebbe fallire per mancanza di risorse
