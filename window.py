from PySide6.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
import sys
import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from modelAGS import Model

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_main_layout()
        self.setMinimumSize(600,400)
        self.load_models()

    def setup_main_layout(self):
        self.setWindowTitle("Sales prediction model")

        layout = QVBoxLayout()

        buttonAGS = QPushButton("Model based on AGS value")
        buttonShafty = QPushButton("Model based on Shaft value")
        buttonSitling = QPushButton("Model based on Sitlingi value")

        buttonAGS.clicked.connect(lambda: self.show_input_window("AGS"))
        buttonShafty.clicked.connect(lambda: self.show_input_window("Shafty"))
        buttonSitling.clicked.connect(lambda: self.show_input_window("Sitlingi"))

        layout.addWidget(buttonAGS)
        layout.addWidget(buttonShafty)
        layout.addWidget(buttonSitling)

        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)
    
    def show_input_window(self, model):
        
        inside_layout = QVBoxLayout()
        self.setWindowTitle(f"Model based on {model} value")

        backBtn = QPushButton("Back")
        inside_layout.addWidget(backBtn)
        backBtn.clicked.connect(self.reset_to_main)

        sub_layouts = self.set_layout(model)
        for sub_layout in sub_layouts:
            inside_layout.addLayout(sub_layout)

        predictBtn = QPushButton('Ok')
        predictBtn.clicked.connect(lambda: self.predict(model))
        inside_layout.addWidget(predictBtn)

        inside_container = QWidget()
        inside_container.setLayout(inside_layout)
        self.setCentralWidget(inside_container)

    def reset_to_main(self):
        self.setup_main_layout()

    def set_layout(self, model):
        l = []

        if model == "AGS":
            sub_layout_1 = QHBoxLayout()
            AGSlabel = QLabel("AGS:")
            self.AGSinput = QLineEdit()
            sub_layout_1.addWidget(AGSlabel)
            sub_layout_1.addWidget(self.AGSinput)
            l.append(sub_layout_1)

            sub_layout_2 = QHBoxLayout()
            ShaftyLabel = QLabel("Shafty:")
            self.ShaftyValueLabel = QLabel("placeholder")
            sub_layout_2.addWidget(ShaftyLabel)
            sub_layout_2.addWidget(self.ShaftyValueLabel)
            l.append(sub_layout_2)

            sub_layout_3 = QHBoxLayout()
            SitlingiLabel = QLabel("Sitlingi:")
            self.SitlingiValueLabel = QLabel("placeholder")
            sub_layout_3.addWidget(SitlingiLabel)
            sub_layout_3.addWidget(self.SitlingiValueLabel)
            l.append(sub_layout_3)

        elif model == "Shafty":
            sub_layout_1 = QHBoxLayout()
            ShaftyLabel = QLabel("Shafty:")
            ShaftyInput = QLineEdit()
            sub_layout_1.addWidget(ShaftyLabel)
            sub_layout_1.addWidget(ShaftyInput)
            l.append(sub_layout_1)

            sub_layout_2 = QHBoxLayout()
            AGSLabel = QLabel("AGS:")
            AGSValueLabel = QLabel("placeholder")
            sub_layout_2.addWidget(AGSLabel)
            sub_layout_2.addWidget(AGSValueLabel)
            l.append(sub_layout_2)

            sub_layout_3 = QHBoxLayout()
            SitlingiLabel = QLabel("Sitlingi:")
            SitlingiValueLabel = QLabel("placeholder")
            sub_layout_3.addWidget(SitlingiLabel)
            sub_layout_3.addWidget(SitlingiValueLabel)
            l.append(sub_layout_3)

        elif model == "Sitlingi":
            sub_layout_1 = QHBoxLayout()
            SitlingiLabel = QLabel("Sitlingi:")
            SitlingiInput = QLineEdit()
            sub_layout_1.addWidget(SitlingiLabel)
            sub_layout_1.addWidget(SitlingiInput)
            l.append(sub_layout_1)

            sub_layout_2 = QHBoxLayout()
            AGSLabel = QLabel("AGS:")
            AGSValueLabel = QLabel("placeholder")
            sub_layout_2.addWidget(AGSLabel)
            sub_layout_2.addWidget(AGSValueLabel)
            l.append(sub_layout_2)

            sub_layout_3 = QHBoxLayout()
            ShaftyLabel = QLabel("Shafty:")
            ShaftyValueLabel = QLabel("placeholder")
            sub_layout_3.addWidget(ShaftyLabel)
            sub_layout_3.addWidget(ShaftyValueLabel)
            l.append(sub_layout_3)

        return l
    
    def load_models(self):
        self.modelAGS = Model()
        self.modelAGS.load_state_dict(torch.load('modelAGS.pth', weights_only=True))
        self.modelAGS.eval()
        self.scalerAGS = joblib.load('scalerAGS.pkl')

        self.modelShafty = Model()
        self.modelShafty.load_state_dict(torch.load('modelShafty.pth', weights_only=True))
        self.modelShafty.eval()
        self.scalerShafty = joblib.load('scalerShafty.pkl')

        self.modelSitlingi = Model()
        self.modelSitlingi.load_state_dict(torch.load('modelSitlingi.pth', weights_only=True))
        self.modelSitlingi.eval()
        self.scalerSitlingi = joblib.load('scalerSitlingi.pkl')
    
    def predict(self, model):
        if model == "AGS":
            input_value = float(self.AGSinput.text())
            test_values = np.array([[input_value]])
            scaled_test_value = self.scaler.transform(np.hstack([test_values, [[0, 0]]]))[:, 0:1]
            test_tensor = torch.tensor(scaled_test_value, dtype=torch.float32)

            with torch.no_grad():
                scaled_pred = self.model(test_tensor)

            scaled_pred_np = scaled_pred.numpy()
            scaled_pred_dummy = np.hstack([[0], scaled_pred_np[0]])
            og_pred = self.scaler.inverse_transform([scaled_pred_dummy])[0][1:]

            self.ShaftyValueLabel.setText(f"{og_pred[0]:.4f}")
            self.SitlingiValueLabel.setText(f"{og_pred[1]:.4f}")

