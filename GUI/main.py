# main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from main_window import Ui_MainWindow

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == "__main__":
    app = QApplication([])
    main_win = MyMainWindow()
    main_win.show()
    sys.exit(app.exec_())