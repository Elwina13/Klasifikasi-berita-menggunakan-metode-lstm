# -*- coding: utf-8 -*-
from tensorflow import keras
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, string
import subprocess
import os
import sys
import pandas as pd
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QStatusBar, QLineEdit, QVBoxLayout, QWidget, QFileDialog, QTableWidget, QTableWidgetItem)

from ReadDataSheet import DataProcessor

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(611, 372)
        MainWindow.setStyleSheet(u"")
        MainWindow.setWindowTitle("Sistem Klasifikasi Berita")
        self.namaFile = ''
        self.read_datasheet = DataProcessor()
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_7 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)

        self.horizontalLayout_6.addWidget(self.label_2)

        self.btnPilihDatasheet = QPushButton(self.centralwidget)
        self.btnPilihDatasheet.setObjectName(u"btnPilihDatasheet")

        self.horizontalLayout_6.addWidget(self.btnPilihDatasheet)

        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_2.addWidget(self.label_4)

        self.tableWidget_2 = QTableWidget(self.centralwidget)
        self.tableWidget_2.setObjectName(u"tableWidget_2")

        self.verticalLayout_2.addWidget(self.tableWidget_2)

        self.horizontalLayout_4.addLayout(self.verticalLayout_2)

        self.horizontalLayout_3.addLayout(self.horizontalLayout_4)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.btnProcessing = QPushButton(self.centralwidget)
        self.btnProcessing.setObjectName(u"btnProcessing")

        self.horizontalLayout_5.addWidget(self.btnProcessing)

        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_6.addWidget(self.label_5)

        self.tableWidget = QTableWidget(self.centralwidget)
        self.tableWidget.setObjectName(u"tableWidget")

        self.verticalLayout_6.addWidget(self.tableWidget)

        self.horizontalLayout_3.addLayout(self.verticalLayout_6)

        self.verticalLayout_7.addLayout(self.horizontalLayout_3)

        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_10)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(1)
        sizePolicy2.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy2)
        self.label_6.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_7.addWidget(self.label_6)

        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_7.addWidget(self.label_8)

        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.horizontalLayout_7.addWidget(self.lineEdit)

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")

        self.horizontalLayout_7.addWidget(self.pushButton)

        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy3)
        self.label_7.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_8.addWidget(self.label_7)

        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")
        sizePolicy3.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy3)

        self.horizontalLayout_8.addWidget(self.label_9)

        self.text_prediksi = QLabel(self.centralwidget)
        self.text_prediksi.setObjectName(u"text_prediksi")

        self.horizontalLayout_8.addWidget(self.text_prediksi)

        self.verticalLayout.addLayout(self.horizontalLayout_8)

        self.verticalLayout_7.addLayout(self.verticalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btnPilihDatasheet.clicked.connect(self.getFileName)

        self.label_6.setText("Masukkan Kalimat")
        self.label_7.setText("Hasil Prediksi")
        self.label_8.setText(":")
        self.label_9.setText(":")
        self.pushButton.setText("Pengujian")
        self.pushButton.clicked.connect(self.getPredict)
        self.tableWidget_2.resizeColumnToContents(1)

        QMetaObject.connectSlotsByName(MainWindow)


    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Sistem Klasifikasi Berita", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Input data", None))
        self.btnPilihDatasheet.setText(QCoreApplication.translate("MainWindow", u"Input file", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"", None))
        self.label_3.setText("")

        self.btnProcessing.setText(QCoreApplication.translate("MainWindow", u"Classify", None))
        self.btnProcessing.clicked.connect(self.show_eval)
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Hasil Evaluasi Klasifikasi", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"KLASIFIKASI JUDUL BERITA MENGGUNAKAN LSTM", None))

    # retranslateUi
    def show_eval(self):
        file_eval = "aggregated_metrics_datasheet.csv"
        data = pd.read_csv(file_eval)
        num_rows, num_cols = data.shape

        self.tableWidget.setRowCount(num_rows)
        self.tableWidget.setColumnCount(num_cols)
        self.tableWidget.setHorizontalHeaderLabels(data.columns)
        for row in range(num_rows):
            for col in range(num_cols):
                value = data.iloc[row, col]
                if isinstance(value, (int, float)):
                    # Jika nilai adalah angka, tampilkan dengan dua desimal
                    value_str = "{:.2f}".format(value)
                else:
                    # Jika bukan angka, tampilkan apa adanya
                    value_str = str(value)

                item = QTableWidgetItem(value_str)
                self.tableWidget.setItem(row, col, item)
    def getFileName(self):
        file_filter = 'Data File (*.xlsx *.csv);;'


        file_paths, _ = QFileDialog.getOpenFileNames()
        self.namaFile = file_paths

        if file_paths:
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                file_base_name, file_extension = os.path.splitext(file_name)
                print("file name=", file_name)
                print('file_base_name=', file_base_name)
        self.read_datasheet.readDatasheet(file_base_name, file_name, self.tableWidget_2)
    def getPredict(self, name):
        data = pd.read_csv(self.namaFile[0], encoding='latin-1')
        data['category'] = data['category'].str.lower()
        data.drop(data[data.category == 'news'].index, inplace=True)
        data.drop(data[data.category == 'hot'].index, inplace=True)

        max_features = 2000
        max_len = 100

        self.label = pd.Series(data['category']).unique()
        my_file = 'tokenized.npy'

        data = np.load(my_file, allow_pickle=True)
        data = pd.Series(data)
        joined = data.apply(lambda x: ' '.join(x))
        self.tokenizer = Tokenizer(num_words=max_features, split=' ')
        self.tokenizer.fit_on_texts(joined.values)
        X = self.tokenizer.texts_to_sequences(joined.values)
        X = pad_sequences(X, maxlen=max_len)
        self.loaded_model = keras.models.load_model('model_LSTM.h5')
        self.textJudul = self.lineEdit.text()
        self.textJudul = self.cleaning(self.textJudul)
        self.textJudul = self.remove_stopwords(self.textJudul)
        self.prediksiModel(self.textJudul)

    def cleaning(self, text):
        # Case folding
        text = text.lower()
        # Trim text
        text = text.strip()
        # Remove punctuations, special characters, and double whitespace
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        # Number removal
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        # Remove number and whitespaces
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text

    def prediksiModel(self, text):
        text_seq = self.tokenizer.texts_to_sequences([text])
        text_seq = pad_sequences(text_seq, maxlen=100)
        pred = self.loaded_model.predict(text_seq)[0]
        predict = self.label[np.argmax(pred)]
        self.text_prediksi.setText(predict)

    # Fungsi untuk menghapus kata-kata berhenti dari teks
    def remove_stopwords(self, text):
        stop_words = set(stopwords.words("indonesian"))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
