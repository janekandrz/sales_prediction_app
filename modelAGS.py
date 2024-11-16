import torch 
import torch.nn as nn
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import joblib

data = pd.read_csv("data.csv")

df = pd.DataFrame(data)
df = df.replace(r'\s+','',regex=True).astype(int)

scaler= MinMaxScaler()
scaled_data=scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,columns=df.columns)

x = torch.tensor(scaled_df["AGS"].values, dtype=torch.float32).view(-1,1)
y = torch.tensor(scaled_df[["Shafty","Slitingi"]].values, dtype=torch.float32)

learning_rate =0.01
class ModelAGS(nn.Module):
    def __init__(self):
        super(ModelAGS,self).__init__()
        self.hidden = nn.Linear(1, 10) 
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.hidden(x)) 
        x = self.output(x)
        return x


model = ModelAGS()
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

if __name__=="__main__":
    epochs = 500
    for epoch in range(epochs):
        model.train()
    
        predictions = model(x)
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    
    torch.save(model.state_dict(), 'modelAGS.pth')
    joblib.dump(scaler, 'scalerAGS.pkl')
    
    
    