from PySide6.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
import sys
import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from modelAGS import ModelAGS
from modelShafty import ModelShafty
from modelSitlingi import ModelSitlingi as ModelSlitingi


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
        buttonSitling = QPushButton("Model based on Slitingi value")

        buttonAGS.clicked.connect(lambda: self.show_input_window("AGS"))
        buttonShafty.clicked.connect(lambda: self.show_input_window("Shafty"))
        buttonSitling.clicked.connect(lambda: self.show_input_window("Slitingi"))

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
            SlitingiLabel = QLabel("Slitingi:")
            self.SlitingiValueLabel = QLabel("placeholder")
            sub_layout_3.addWidget(SlitingiLabel)
            sub_layout_3.addWidget(self.SlitingiValueLabel)
            l.append(sub_layout_3)

        elif model == "Shafty":
            sub_layout_1 = QHBoxLayout()
            ShaftyLabel = QLabel("Shafty:")
            self.ShaftyInput = QLineEdit()
            sub_layout_1.addWidget(ShaftyLabel)
            sub_layout_1.addWidget(self.ShaftyInput)
            l.append(sub_layout_1)

            sub_layout_2 = QHBoxLayout()
            AGSLabel = QLabel("AGS:")
            self.AGSValueLabel = QLabel("placeholder")
            sub_layout_2.addWidget(AGSLabel)
            sub_layout_2.addWidget(self.AGSValueLabel)
            l.append(sub_layout_2)

            sub_layout_3 = QHBoxLayout()
            SlitingiLabel = QLabel("Slitingi:")
            self.SlitingiValueLabel = QLabel("placeholder")
            sub_layout_3.addWidget(SlitingiLabel)
            sub_layout_3.addWidget(self.SlitingiValueLabel)
            l.append(sub_layout_3)

        elif model == "Slitingi":
            sub_layout_1 = QHBoxLayout()
            SlitingiLabel = QLabel("Slitingi:")
            self.SlitingiInput = QLineEdit()
            sub_layout_1.addWidget(SlitingiLabel)
            sub_layout_1.addWidget(self.SlitingiInput)
            l.append(sub_layout_1)

            sub_layout_2 = QHBoxLayout()
            AGSLabel = QLabel("AGS:")
            self.AGSValueLabel = QLabel("placeholder")
            sub_layout_2.addWidget(AGSLabel)
            sub_layout_2.addWidget(self.AGSValueLabel)
            l.append(sub_layout_2)

            sub_layout_3 = QHBoxLayout()
            ShaftyLabel = QLabel("Shafty:")
            self.ShaftyValueLabel = QLabel("placeholder")
            sub_layout_3.addWidget(ShaftyLabel)
            sub_layout_3.addWidget(self.ShaftyValueLabel)
            l.append(sub_layout_3)

        return l
    
    def load_models(self):
        self.modelAGS = ModelAGS()
        self.modelAGS.load_state_dict(torch.load('modelparams/modelAGS.pth', weights_only=True))
        self.modelAGS.eval()
        self.scalerAGS = joblib.load('scalers/scalerAGS.pkl')

        self.modelShafty = ModelShafty()
        self.modelShafty.load_state_dict(torch.load('modelparams/modelShafty.pth', weights_only=True))
        self.modelShafty.eval()
        self.scalerShafty = joblib.load('scalers/scalerShafty.pkl')

        self.modelSlitingi = ModelSlitingi()
        self.modelSlitingi.load_state_dict(torch.load('modelparams/modelSitlingi.pth', weights_only=True))
        self.modelSlitingi.eval()
        self.scalerSlitingi = joblib.load('scalers/scalerSitlingi.pkl')
    
    def predict(self, model):
        if model == "AGS":
            input_value = float(self.AGSinput.text())
            test_values = np.array([[input_value]])
            scaled_test_value = self.scalerAGS.transform(np.hstack([test_values, [[0, 0]]]))[:, 0:1]
            test_tensor = torch.tensor(scaled_test_value, dtype=torch.float32)

            with torch.no_grad():
                scaled_pred = self.modelAGS(test_tensor)

            scaled_pred_np = scaled_pred.numpy()
            scaled_pred_dummy = np.hstack([[0], scaled_pred_np[0]])
            og_pred = self.scalerAGS.inverse_transform([scaled_pred_dummy])[0][1:]

            self.ShaftyValueLabel.setText(f"{og_pred[0]:.4f}")
            self.SlitingiValueLabel.setText(f"{og_pred[1]:.4f}")
        elif model == "Shafty":
            input_value = float(self.ShaftyInput.text())
            test_values = np.array([[input_value]])
            scaled_test_value = self.scalerShafty.transform(np.hstack([test_values, [[0, 0]]]))[:, 0:1]
            test_tensor = torch.tensor(scaled_test_value, dtype=torch.float32)

            with torch.no_grad():
                scaled_pred = self.modelShafty(test_tensor)

            scaled_pred_np = scaled_pred.numpy()
            scaled_pred_dummy = np.hstack([[0], scaled_pred_np[0]])
            og_pred = self.scalerShafty.inverse_transform([scaled_pred_dummy])[0][1:]

            self.AGSValueLabel.setText(f"{og_pred[0]:.4f}")
            self.SlitingiValueLabel.setText(f"{og_pred[1]:.4f}")
        elif model == "Slitingi":
            input_value = float(self.SlitingiInput.text())
            test_values = np.array([[input_value]])
            scaled_test_value = self.scalerSlitingi.transform(np.hstack([test_values, [[0, 0]]]))[:, 0:1]
            test_tensor = torch.tensor(scaled_test_value, dtype=torch.float32)

            with torch.no_grad():
                scaled_pred = self.modelSlitingi(test_tensor)

            scaled_pred_np = scaled_pred.numpy()
            scaled_pred_dummy = np.hstack([[0], scaled_pred_np[0]])
            og_pred = self.scalerSlitingi.inverse_transform([scaled_pred_dummy])[0][1:]

            self.AGSValueLabel.setText(f"{og_pred[0]:.4f}")
            self.ShaftyValueLabel.setText(f"{og_pred[1]:.4f}")
