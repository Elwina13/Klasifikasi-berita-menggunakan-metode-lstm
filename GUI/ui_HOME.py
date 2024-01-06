# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'HOMEfTfKGQ.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import nltk
# Tokenizing
nltk.download('punkt')
# Stopwords removal
nltk.download('stopwords')
# Lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

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
from sklearn.preprocessing import LabelEncoder



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(611, 410)
        MainWindow.setStyleSheet(u"")
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

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"KLASIFIKASI JUDUL BERITA MENGGUNAKAN LSTM", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.btnPilihDatasheet.setText(QCoreApplication.translate("MainWindow", u"PILIH DATASHEET", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"TAMPILAN DATA AWAL", None))
        self.label_3.setText("")
        self.btnProcessing.setText(QCoreApplication.translate("MainWindow", u"RUN PROCESSING", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"TAMPILAN DATA SETELAH PROCESSING", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"UJI MODEL LSTM", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Masukkan Kalimat ", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u":", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Pengujian", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Hasil Prediksi", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u":", None))
        self.text_prediksi.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
    # retranslateUi

