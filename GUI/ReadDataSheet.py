# data_processor.py
import nltk
import pandas as pd
from tqdm import tqdm
import re, string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# nltk.download('punkt') # Tokenizing
# nltk.download('stopwords') # Stopwords removal
# nltk.download('wordnet')# Lemmatizer
# nltk.download('omw-1.4')
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

class DataProcessor:
    def readDatasheet(self, ekstensi='csv', namaFile='', widget=''):
        tqdm.pandas()

        data = pd.read_csv(namaFile)
        num_rows, num_cols = data.shape

        widget.setRowCount(1)
        widget.setColumnCount(1)
        widget.setHorizontalHeaderLabels(["Nama File"])
        item = QTableWidgetItem(namaFile)
        widget.setItem(0, 0, item)

