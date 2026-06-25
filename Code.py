import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from google.colab import files
import statsmodels.api as sm
import gradio as gr

csv = "https://raw.githubusercontent.com/AnushaMahajan12/Canopyt/refs/heads/main/ShadeData.csv"
tree_data = pd.read_csv(csv)

#Creates new column for what lack of canopy cover has to be for MasterScore = 0.25
canCovDif = []
for index, row in tree_data.iterrows(): #1.5 = 0.25*6 = average MasterScore excluding the Canopy Cover --> subtracting all the indices = difference of canopy needed to make city reach threshold sustainability
  canopyDifferences = 1.5 - (row['TrafficProximityIndex'] + row['PopDensityIndex'] + row['AirQualityIndex'] + row['LowIncomeIndex'] + row['HeatImpervIndex'])
  canCovDif.append(canopyDifferences)
tree_data['CanopyCoverDifference'] = canCovDif 

#Updates csv to add column of difference
tree_data.to_csv('updated_tree_data.csv', index=False)
#files.download('updated_tree_data.csv')
model_data = pd.read_csv('updated_tree_data.csv')

model_data = model_data.dropna(subset=['TrafficProximityIndex', 'AirQualityIndex', 'LowIncomeIndex', 'CanopyCoverDifference']) #chosen indices drop any values that don't exist
model_data.to_csv('final_tree_data.csv', index=False)
#files.download('final_tree_data.csv')

X = model_data[['TrafficProximityIndex', 'AirQualityIndex', 'LowIncomeIndex']] #inputs
y = model_data['CanopyCoverDifference'] #output

#need a constant --> linear regression model = mx + !b!
X = sm.add_constant(X)
model = sm.OLS(y, X).fit() #fits model 

#print(model.summary())
#directly related to Canopy Cover Difference --> very low coef (for one unit of Index --> drops by that much for CCD)
#Durbin Watson --> explains how data is clustered (one area with low score = next to another with low score)

def predictCan(TrafficProximityIndex, AirQualityIndex, LowIncomeIndex): #define predict as input 3 indices
  input = pd.DataFrame([{
      'TrafficProximityIndex': float(TrafficProximityIndex),
      'AirQualityIndex': float(AirQualityIndex),
      'LowIncomeIndex': float(LowIncomeIndex)
      }])
  input.insert(0, 'constant', 1.0) #acts as constant (need to add other dimension due to linear constant)
  canopyCover = model.predict(input)[0] #model predict cover w/ input
  return f'Predicted Canopy Cover Difference: {canopyCover: .2f}'

interface = gr.Interface( #creates interface
    fn = predictCan, #predict the canopy cover dif w/ sliders for inputs all in order
    inputs = [
        gr.Slider(minimum = 0.00, maximum = 1.00, value = 0.50, label = "Traffic Proximity Index"),
        gr.Slider(minimum = 0.00, maximum = 1.00, value = 0.50, label = "Air Quality Index"),
        gr.Slider(minimum = 0.00, maximum = 1.00, value = 0.50, label = "Low Income Index")
    ],
    outputs = gr.Textbox(label = "Model Prediction"), #labels on interface
    title = "Canopy Need Predictor",
    description = "Adjust the different indices to fit the city to determine a prediction for how much canopy cover is needed to make the city more sustainable."
)

interface.launch(share = True)
