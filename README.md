# Sentiment-Analysis-Using-LSTM-and-DistilBERT-Models


---

# Sentiment Analysis on Yelp Restaurant Reviews Using DistilBERT and LSTM

## Overview
This project performs sentiment analysis on Yelp restaurant reviews using:
1. **DistilBERT** (transformer-based model)
2. **LSTM** (recurrent neural network with Optuna for hyperparameter tuning)

Sentiments are classified into three categories: negative, neutral, and positive. Both models are fine-tuned to optimize performance metrics such as accuracy, F1 score, precision, and recall.

---

## Prerequisites
Ensure you have the following installed:
- Python 3.7+
- PyTorch, Transformers (for DistilBERT)
- TensorFlow, Keras, Optuna (for LSTM)
- Scikit-learn, Matplotlib, Pandas, NumPy

Install dependencies:
```bash
pip install torch transformers tensorflow optuna scikit-learn matplotlib pandas numpy
```

---

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-yelp-reviews.git
   cd sentiment-analysis-yelp-reviews
   ```
2. **Prepare the Dataset**: Download Yelp reviews in CSV format with columns for Yelp URL, Rating, Date, and Review Text. Update the `file_path` variable in the script to the location of your dataset.
3. **Set up Python Environment**: If using a local environment, set up and activate a virtual environment, then install the necessary packages.

---

## Running the Program
1. **Run the Script**: Run directly in Python or a Jupyter notebook. Colab users can mount Google Drive if necessary.
2. **Training and Evaluation**: Train the DistilBERT model using the Hugging Face Trainer API and the LSTM model with hyperparameters tuned using Optuna.
3. **Generate Results**: Visualize metrics like accuracy, F1 score, precision, recall, confusion matrix, and loss curves.

---

## Additional Notes
- The script automatically checks for GPU availability for faster computation.
- Trained models (DistilBERT and LSTM) are saved for future inference.
