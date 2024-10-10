# Disaster vs. No Disaster Classification Using NLP

<img src="https://assets-lbmjournal-com.s3.us-east-2.amazonaws.com/2023/09/PLM-Featured-Image-696x392.jpg">

This project uses Natural Language Processing (NLP) techniques to classify tweets or text data as either indicating a **disaster** or **no disaster**. It is built using Python and deep learning frameworks to train models for this classification task.

## Project Overview

This project aims to build a model that can automatically classify text data as relevant to a disaster or not. We leverage natural language processing and deep learning models to analyze the data and predict whether a text (tweet) is related to a disaster.

## Dataset

The dataset used in this project consists of labeled examples of tweets. Each tweet is categorized as:
- **1**: Relevant to a disaster
- **0**: Not relevant to a disaster

The dataset is split into training and test sets and can be found in `data/train.csv` and `data/test.csv`.

## Modeling

We use the following steps for modeling:

1. **Data Preprocessing**: 
   - The data is cleaned and tokenized to remove noise and prepare it for model input.
   - Tokenization and vectorization techniques like TF-IDF or embeddings are employed.

2. **Deep Learning Models**: 
   - Various deep learning architectures are explored, such as Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, Gated Recurrent Unit Networks (GRU), 1D Convolutional Neural Networks (CNNs) and a [pre-trained model](https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2) tailored for text classification.
   - Models are trained on the labeled dataset and evaluated using accuracy, precision, recall, and F1-score metrics.

3. **Visualization**:
   - Loss curves and accuracy plots are visualized to track model performance during training.
   - TensorBoard is used for visualizing model metrics.

## Requirements

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `tensorflow`
- `matplotlib`
- `sklearn`

You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/iamharshvardhan/disaster-classification
    ```

2. **The dataset** is in the `data/` folder.

3. **Run the notebook**:
   - Open the Jupyter Notebook (`disaster-no-disaster.ipynb`) and run the cells to preprocess data, train models, and visualize results.
   - You can also modify the model architectures in the notebook.

4. **Training**: 
   - Run the notebook's training cells to train the model on the provided dataset.

## Results

The model achieves competitive performance in classifying disaster-related tweets. You can track the results and metrics (such as accuracy, precision, recall, and F1-score) via the visualizations produced during training. The callbacks for each model is saved in the [model_logs](model_logs) folder.

## Acknowledgements

- The dataset is provided by [kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data).
- This project uses techniques inspired by various NLP research and tutorials in deep learning.
