<div align="center"> 
  <img src="LiricsFinder_logo.webp" alt="Logo LiricsFinder" width="200" height="200"/>
</div>

<h1 align="center">LiricsFinder</h1>

# 🎵 Lyrics Finder: NLP & BERT per l'Analisi dei Testi Musicali

## 📌 Descrizione del Progetto

Lyrics Finder è un sistema di classificazione automatica dei generi musicali basato sull'analisi testuale delle canzoni, utilizzando tecniche avanzate di **Natural Language Processing (NLP)** e il modello **BERT**.

L'idea nasce dalla necessità di categorizzare automaticamente i brani in base al loro testo, senza dover ricorrere a metadati o informazioni aggiuntive. Questo approccio può avere applicazioni in ambiti come la creazione di playlist personalizzate, l'analisi tematica e la scoperta musicale.

Tuttavia, analizzare testi musicali non è un compito semplice: l'uso di metafore, giochi di parole e linguaggio informale rende difficile per i modelli tradizionali ottenere risultati accurati. Per questo motivo, abbiamo scelto **BERT**, un modello avanzato di deep learning capace di comprendere il contesto in modo bidirezionale.

---

## 📂 Dataset

Il dataset utilizzato contiene le seguenti informazioni:

- **Index**
- **Titolo della canzone** 🎵
- **Anno di pubblicazione** 📅
- **Artista** 🎤
- **Genere musicale** 🎼
- **Testo della canzone** ✍️

### 🔍 Exploratory Data Analysis (EDA)

- **Totale campioni**: 362.237
- **Generi più rappresentati**: Rock (131.377), Pop (49.444)
- **Generi meno rappresentati**: Indie (5.732), R&B (5.935), Folk (3.241)
- **Problema**: squilibrio tra le classi

---

## 🔧 Pipeline

### 🧼 1. Cleaning

- Rimozione di dati nulli o non disponibili
- Eliminazione di punteggiatura e simboli superflui
- Pulizia dei testi strumentali o corrotti
- Bilanciamento del dataset tramite **undersampling**

### ⚙️ 2. Preprocessing

- Rimozione delle **stop words**
- **Lemmatizzazione**
- **Encoding** dei generi
- **Splittaggio** del dataset
- **Tokenizzazione**

### 🤖 3. Modeling

- Utilizzo del modello **BERT pre-addestrato** (**BertForSequenceClassification**, *bert-base-uncased*)

### 🚀 4. Training

- **Early Stopping** (2 epoche)
- **Forward pass** → **Calcolo perdita** → **Backpropagation** → **Aggiornamento pesi**
- **Interruzione alla 4ª epoca** per evitare overfitting

### 📊 5. Performance Evaluation

- **Matrice di Confusione**: buona accuratezza, ma difficoltà nella distinzione tra Rock e Hip-Hop
- **Classification Report**:
  - 🎸 **Pop & Metal**: Buoni risultati
  - 🎤 **Rock & Hip-Hop**: Performance più basse
- **Accuratezza complessiva**: **71%**

---

## 🔥 Ottimizzazione

### 🛠 Data Augmentation

- **Back Translation**
- **Synonym Replacement**
- **Miglioramento dell'accuratezza al 73%**

### 🎯 Ulteriori Tecniche

- **Aumento della lunghezza massima** dei token
- **Riduzione del learning rate**
- **Dropout**
- **Focal Loss**
- **Accuracy finale**: **72%**

---

## 🔮 Lavori Futuri

- 📈 **Espansione del dataset**
- 🖥️ **Utilizzo di hardware più potente**
- 🔄 **Testare nuove tecniche di data augmentation**
- 🏆 **Sperimentare altri modelli BERT**
- 🔧 **Ottimizzazione con hyperparameter tuning**

---

## ⚙️ Esecuzione

Lyrics Finder è progettato per essere eseguito in due modalità:

- **💻 Esecuzione locale**: per utenti con **GPU NVIDIA** (RTX 3060, 3070, 3080 o superiori)
- **☁️ Google Colab**: per chi non dispone di risorse hardware avanzate

---

## 📦 Requisiti

- **Python 3.x** 🐍
- **PyTorch** 🔥
- **Transformers** (Hugging Face)
- **Pandas, NumPy, Scikit-learn**
- **Google Colab** (opzionale)

---

## 🚀 Come Usare il Progetto

### Clonare il repository:
```bash
git clone https://github.com/tuo-utente/LyricsFinder.git
cd LyricsFinder
```



