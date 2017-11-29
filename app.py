import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

import main_driver


# This creates a GUI to run the program
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Choose a Question'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()
        self.button_1 = QPushButton('Question 1', self)
        self.button_2 = QPushButton('Question 2', self)
        self.button_3 = QPushButton('Question 3', self)
        self.button_4 = QPushButton('Question 4', self)
        self.button_5 = QPushButton('Question 5', self)

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # button for question 1
        self.button_1.setToolTip('Question 1 button')
        self.button_1.move(0, 0)

        # button for question
        self.button_2.setToolTip('Question 2 button')
        self.button_2.move(100, 70)

        # button for question 3
        self.button_3.setToolTip('Question 3 button')
        self.button_3.move(100, 70)

        # button for question 4
        self.button_4.setToolTip('Question 4 button')
        self.button_4.move(100, 70)

        # button for question 5
        self.button_5.setToolTip('Question 5 button')
        self.button_5.move(100, 70)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    App.button_1.clicked.connect(main_driver.query_1())
    App.button_2.clicked.connect(main_driver.query_2())
    App.button_3.clicked.connect(main_driver.query_3())
    App.button_4.clicked.connect(main_driver.query_4())
    App.button_5.clicked.connect(main_driver.query_5())

    sys.exit(app.exec_())
