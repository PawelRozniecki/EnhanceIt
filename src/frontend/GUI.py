from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QHBoxLayout, QWidget,QVBoxLayout
from PyQt5.QtGui import QPixmap
import  multiprocessing as mp
import sys
import  os
import  time

sys.path.append('/Users/pingwin/PycharmProjects/EnhanceIt/')
from src.test_image import main


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        widget = QWidget()

        vbox = QVBoxLayout()


        layout = QHBoxLayout()

        self.label = QLabel()
        self.label2 = QLabel()
        layout.addWidget(self.label)
        layout.addWidget(self.label2)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.initUI()

    def initUI(self):
        self.setGeometry(500, 500, 800, 500)
        self.setWindowTitle("EnhanceIT")
        button = QPushButton('Choose a file', self)
        button.move(100, 7)
        button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        self.openFileWindow()

    def openFileWindow(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            path = fileName
            print(path)
            pixmap = QPixmap(fileName)
            x, y = pixmap.width(), pixmap.height()
            self.label.setPixmap(pixmap)
            p = mp.Process(target=enhance, args=(path,))
            p.start()
            print(p.is_alive())
            p.join()
            print(p.is_alive())

            print("FINISHED")
            pxmap2 = QPixmap('/Users/pingwin/PycharmProjects/EnhanceIt/src/frontend/savename.png')
            self.label2.setPixmap(pxmap2.scaled(x, y))
            self.resize(pixmap.size())
            self.adjustSize()


def enhance(filepath):
    main(filepath)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())