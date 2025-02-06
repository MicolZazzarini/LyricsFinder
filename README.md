<div align="center"> 
  <img src="LiricsFinder_logo.webp" alt="LiricsFinder Logo" width="200" height="200"/>
</div>

<h1 align="center">LiricsFinder</h1>

# 🎵 Lyrics Finder: NLP & BERT for Music Lyrics Analysis

## 📌 Project Description

Lyrics Finder is an automatic music genre classification system based on the textual analysis of song lyrics, using advanced **Natural Language Processing (NLP)** techniques and the **BERT** model.

The idea arose from the need to automatically categorize songs based on their lyrics, without relying on metadata or additional information. This approach has applications in areas such as personalized playlist creation, thematic analysis, and music discovery.

However, analyzing music lyrics is not a simple task: the use of metaphors, wordplay, and informal language makes it difficult for traditional models to achieve accurate results. For this reason, we chose **BERT**, an advanced deep learning model capable of understanding context in a bidirectional manner.

---

## 📂 Project Structure

The project is organized into the following folders:

### 📁 `notebooks/`

This folder contains executable scripts for **Google Colab**, developed to experiment with various stages of the pipeline:

- **`data_augmentation.ipynb`**: Pipeline with **data augmentation** techniques.
- **`first_pipeline.ipynb`**: Initial pipeline developed **without data augmentation** or optimization.
- **`optimization.ipynb`**: **Optimized pipeline** to improve model performance.

### 📁 `src/`

This folder contains the **final and optimized** version of the pipeline, developed for **local execution**:

- **`cleaning.py`**: Data cleaning script.
- **`eda.py`**: Exploratory data analysis of the dataset.
- **`preprocessing.py`**: Transformation and preparation of text for the model.
- **`modeling.py`**: Model creation and configuration using **BERT**.
- **`training.py`**: Model training.
- **`performance_evaluation.py`**: Model performance evaluation.
- **`pipeline.py`**: Main script that runs the **entire pipeline** sequentially, from data cleaning to final evaluation.
- **`requirements.txt`**: File with dependencies needed to run the project locally.

Users can choose to **run each stage separately** or start the entire process by executing `pipeline.py`.

---

## 📂 Dataset

The dataset used contains the following information:

- **Index**
- **Song Title** 🎵
- **Year of Release** 📅
- **Artist** 🎤
- **Music Genre** 🎼
- **Song Lyrics** ✍️

### 🔍 Exploratory Data Analysis (EDA)

- **Total samples**: 362,237
- **Most represented genres**: Rock (131,377), Pop (49,444)
- **Least represented genres**: Indie (5,732), R&B (5,935), Folk (3,241)
- **Problem**: Class imbalance

---

## 🔧 Pipeline

### 🧼 1. Cleaning

- Removal of null or missing data
- Elimination of unnecessary punctuation and symbols
- Cleaning of instrumental or corrupted texts
- Dataset balancing via **undersampling**

### ⚙️ 2. Preprocessing

- Removal of **stop words**
- **Lemmatization**
- **Encoding** of genres
- **Dataset splitting**
- **Tokenization**

### 🤖 3. Modeling

- Use of the **pre-trained BERT model** (**BertForSequenceClassification**, *bert-base-uncased*)

### 🚀 4. Training

- **Early Stopping** (2 epochs)
- **Forward pass** → **Loss calculation** → **Backpropagation** → **Weight update**
- **Stop at 4th epoch** to avoid overfitting

### 📊 5. Performance Evaluation

- **Confusion Matrix**: good accuracy, but difficulties distinguishing between Rock and Hip-Hop
- **Classification Report**:
  - 🎸 **Pop & Metal**: Good results
  - 🎤 **Rock & Hip-Hop**: Lower performance
- **Overall accuracy**: **71%**

---

## 🔥 Optimization

### 🛠 Data Augmentation

- **Back Translation**
- **Synonym Replacement**
- **Improvement in accuracy to 73%**

### 🎯 Additional Techniques

- **Increase of maximum token length**
- **Lower learning rate**
- **Dropout**
- **Focal Loss**
- **Final accuracy**: **72%**

---

## 🔮 Future Work

- 📈 **Expand the dataset**
- 🖥️ **Use more powerful hardware**
- 🔄 **Test new data augmentation techniques**
- 🏆 **Experiment with other BERT models**
- 🔧 **Optimization through hyperparameter tuning**

---

## ⚙️ Execution

Lyrics Finder is designed to run in two modes:

- **💻 Local Execution**: for users with **NVIDIA GPUs** (RTX 3060, 3070, 3080, or higher)
- **☁️ Google Colab**: for users without advanced hardware resources

---

## 📦 Requirements

- **Python 3.x** 🐍
- **PyTorch** 🔥
- **Transformers** (Hugging Face)
- **Pandas, NumPy, Scikit-learn**
- **Google Colab** (optional)

---

# 🚀 How to Use the Project

## ☁️ On Google Colab

1. **Open the desired notebook**
   - Upload one of the `.ipynb` files found in the `notebooks/` folder to Google Colab.
   - For example, open `optimization.ipynb` to run the optimized pipeline.

2. **Connect the runtime to a GPU**
   - Go to `Runtime → Change runtime type → Select GPU`.

3. **Run the cells sequentially**
   - Follow the order of the cells to execute the various stages of the pipeline.

## 💻 Locally

1. **Clone the repository**
```bash
git clone https://github.com/tuo-utente/LyricsFinder.git
cd LyricsFinder/src
```

2. **Install the dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the full pipeline**
```bash
python pipeline.py
```

4. **Run a specific phase of the pipeline (optional)**
```bash
python cleaning.py  # Esegue solo la pulizia dei dati
python training.py  # Addestra il modello
```

---
## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributors

- [Zazzarini Micol](https://github.com/MicolZazzarini)
- [Fiorani Andrea](https://github.com/125ade)
- [Antonini Antonio](https://github.com/tava99)




