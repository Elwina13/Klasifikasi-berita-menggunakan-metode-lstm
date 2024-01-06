from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import nltk
import pandas as pd
from tqdm import tqdm
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class PreProcessing:
    def __init__(self, data_path='indonesian-news-title.csv', max_features=2000, max_len=100, k=10, val_epochs=10):
        self.data_path = data_path
        self.max_features = max_features
        self.max_len = max_len
        self.k = k
        self.val_epochs = val_epochs
        self.label_encoder = LabelEncoder()
        self.tokenizer = Tokenizer(num_words=max_features, split=' ')
        self.evaluations = []
        self.batch_size = 64
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        self.vectorizer = None  # Tambahkan ini untuk inisialisasi atribut vectorizer

    def cleaning(self, text):
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def case_folding(self, text):
        return text.lower()

    def normalize(self, text):
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        return text

    def preproses(self, text):
        text = self.cleaning(text)
        text = self.case_folding(text)
        text = self.normalize(text)
        return text

    def prosesData(self):

        tqdm.pandas()
        data = pd.read_csv(self.data_path)

        data = data.drop(columns=['url', 'date'], axis=1)

        data = data.dropna(axis=0)

        data.drop(data[data.category == 'news'].index, inplace=True)
        data.drop(data[data.category == 'hot'].index, inplace=True)
        data.drop(data[data.category == 'inet'].index, inplace=True)


        data.drop(data[data['title'].duplicated()].index, inplace=True)

        y = self.label_encoder.fit_transform(data['category'])
        max_features = 2000
        max_len = 100

        data['cleaning_titles'] = data['title'].apply(lambda x: self.preproses(x))
        data['tokens'] = data['cleaning_titles'].apply(lambda x: word_tokenize(x))

        data['tokenisasi'] = np.load('tokenized.npy', allow_pickle=True)
        data['joined_tokenisasi'] = data['tokenisasi'].apply(lambda x: ' '.join(x))

        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        X_tfidf = self.vectorizer.fit_transform(data['joined_tokenisasi'].values)

        # Split data
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

        train_label_counts = pd.Series(y_train).value_counts()
        train_label_categories = self.label_encoder.inverse_transform(train_label_counts.index)


        print("Jumlah data Training:")
        for category, count in zip(train_label_categories, train_label_counts):
            print(f"{category} = {count}")


        test_label_counts = pd.Series(y_test).value_counts()
        test_label_categories = self.label_encoder.inverse_transform(test_label_counts.index)


        print("\nJumlah Data Testing:")
        for category, count in zip(test_label_categories, test_label_counts):
            print(f"{category} = {count}")

            # Split data
            indices_train, indices_test = train_test_split(data.index, test_size=0.3, random_state=42)

            # Ambil data asli untuk training dan testing berdasarkan indeks
            data_train = data.loc[indices_train]
            data_test = data.loc[indices_test]


            # Menyimpan data training dan testing ke dalam file CSV
            data_train.to_csv("data_train.csv", index=False)
            data_test.to_csv("data_test.csv", index=False)

        # Set X_train and X_test
        self.X_train = X_train_tfidf
        self.X_test = X_test_tfidf
        self.label_encoder.fit(data['category'])
        data.to_csv("datasheetbaru.csv", index=False)
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(data['joined_tokenisasi'].values)
        X = tokenizer.texts_to_sequences(data['joined_tokenisasi'].values)
        X = pad_sequences(X, maxlen=max_len)
        Y = pd.get_dummies(data['category']).values
        label = pd.Series(data['category']).unique()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        model = self.create_model()
        self.evaluate_model(model, self.X_train, self.y_train)
        model.save("model_LSTM.h5")

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.max_features, 64, input_length=None))
        # Arsitektur LSTM
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    def train_model(self, X_train, y_train, X_test, y_test):
        model = self.create_model()
        history = model.fit(X_train, y_train, epochs=self.val_epochs, batch_size=self.batch_size, validation_data=(X_test, y_test))
        plt.figure(figsize=(18, 6))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.xticks(range(0, 11),)  # Angka 1 hingga 10 sesuai dengan jumlah epoch
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('accuracy_new.png')
        plt.show()
        return model

    def evaluate_model(self, model, X, y):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []
        fold_results = []
        fold_confusion_matrices = []

        for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]


            model = self.train_model(X_train, y_train, X_val, y_val)

            y_pred_prob = model.predict(X_val)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_val, axis=1)

            # Menambahkan perhitungan True Positive, False Positive, True Negative, False Negative
            cm = confusion_matrix(np.argmax(y_val, axis=1), y_pred)

            # Menambahkan perhitungan matriks konfusi untuk setiap label
            num_labels = len(self.label_encoder.classes_)
            for label_idx in range(num_labels):
                class_name = self.label_encoder.classes_[label_idx]
                tp = cm[label_idx, label_idx]
                fp = np.sum(cm[:, label_idx]) - tp
                fn = np.sum(cm[label_idx, :]) - tp
                tn = np.sum(cm) - (tp + fp + fn)

                fold_result = {
                    'Fold': fold,
                    'Class': class_name,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'TN': tn
                }
                fold_results.append(fold_result)

                print(f"Fold {fold} - Class: {class_name}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            # Menghitung metrik evaluasi klasifikasi
            accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred)
            precision = precision_score(np.argmax(y_val, axis=1), y_pred, average='weighted')
            recall = recall_score(np.argmax(y_val, axis=1), y_pred, average='weighted')
            f1 = f1_score(np.argmax(y_val, axis=1), y_pred, average='weighted')

            print(f"Fold {fold} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1_scores.append(f1)

        # Menghitung rata-rata dan menampilkan hasil evaluasi klasifikasi
        mean_accuracy = np.mean(fold_accuracies)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)
        mean_f1 = np.mean(fold_f1_scores)

        print("\nMean Across Folds - Classification Results:")
        print("Mean Accuracy:", mean_accuracy)
        print("Mean Precision:", mean_precision)
        print("Mean Recall:", mean_recall)
        print("Mean F1 Score:", mean_f1)

        # Membuat DataFrame untuk menyimpan hasil evaluasi klasifikasi
        classification_results_df = pd.DataFrame({
            'Fold': range(1, self.k + 1),
            'Accuracy': fold_accuracies,
            'Precision': fold_precisions,
            'Recall': fold_recalls,
            'F1 Score': fold_f1_scores
        })

        # Menyimpan DataFrame hasil evaluasi klasifikasi ke dalam CSV
        classification_results_df.to_csv("aggregated_metrics_datasheet.csv", index=False)

        # Membuat DataFrame untuk menyimpan hasil evaluasi matriks konfusi
        confusion_matrix_results_df = pd.DataFrame(fold_results)

        # Menyimpan DataFrame hasil evaluasi matriks konfusi ke dalam CSV
        confusion_matrix_results_df.to_csv("confusion_matrices_datasheet.csv", index=False)

        # Menyimpan hasil evaluasi ke dalam atribut evaluations
        self.evaluations = {
            'Mean Accuracy': mean_accuracy,
            'Mean Precision': mean_precision,
            'Mean Recall': mean_recall,
            'Mean F1 Score': mean_f1,
            'Classification Results DataFrame': classification_results_df,
            'Confusion Matrix Results DataFrame': confusion_matrix_results_df
        }

        # Setelah selesai semua fold, hitung TP, FP, TN, FN untuk setiap label
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []


        for i, label in enumerate(self.label_encoder.classes_):

            indices = np.where(y_true == i)[0]


            if len(indices) > 100:
                indices = np.random.choice(indices, size=100, replace=False)


            tp = np.sum(y_pred[indices] == i)
            fp = np.sum((y_pred[indices] != i) & (y_true[indices] == i))
            fn = np.sum((y_pred[indices] == i) & (y_true[indices] != i))
            tn = np.sum((y_pred[indices] != i) & (y_true[indices] != i))


            tp_list.append(tp)
            fp_list.append(fp)
            tn_list.append(tn)
            fn_list.append(fn)


        result_df = pd.DataFrame({
            'Class': self.label_encoder.classes_,
            'TP': tp_list,
            'FP': fp_list,
            'FN': fn_list,
            'TN': tn_list
        })


        print(result_df)
        overall_accuracy = accuracy_score(y_true, y_pred)
        overall_precision = precision_score(y_true, y_pred, average='weighted')
        overall_recall = recall_score(y_true, y_pred, average='weighted')
        overall_f1 = f1_score(y_true, y_pred, average='weighted')

        print("\nMetrik evaluasi untuk 500 data")
        print("Overall Accuracy:", overall_accuracy)
        print("Overall Precision:", overall_precision)
        print("Overall Recall:", overall_recall)
        print("Overall F1 Score:", overall_f1)


        overall_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [overall_accuracy, overall_precision, overall_recall, overall_f1]
        })

        overall_metrics_df.to_csv('hasil_evaluasi_500_data.csv', index=False)


        result_df.to_csv('cofusion_500_data.csv', index=False)
        #buat grafik evaluasi

        evaluasi = pd.read_csv('aggregated_metrics_datasheet.csv')
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Accuracy Across Folds')

        # Bar width
        bar_width = 0.1

        # grafik plot
        ax.bar(evaluasi['Fold'] - 1.5 * bar_width, evaluasi['Accuracy'], width=bar_width, label='Accuracy')
        ax.bar(evaluasi['Fold'] - 0.5 * bar_width, evaluasi['Precision'], width=bar_width, label='Precision')
        ax.bar(evaluasi['Fold'] + 0.5 * bar_width, evaluasi['Recall'], width=bar_width, label='Recall')
        ax.bar(evaluasi['Fold'] + 1.5 * bar_width, evaluasi['F1 Score'], width=bar_width, label='F1 Score')

        # Set the y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Set labels and legend
        ax.set_xticks(evaluasi['Fold'])
        ax.set_xticklabels(evaluasi['Fold'])
        ax.set_xlabel('Folds')
        ax.set_ylabel('Metrics Value')
        ax.legend()

        plt.savefig('hasil_evaluasi_bar.png')
        plt.show()

if __name__ == "__main__":
    text_classifier = PreProcessing()
    text_classifier.prosesData()


