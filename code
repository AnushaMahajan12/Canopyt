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
#Split MasterScore into 2 separate lists
threshold_values = []
not_threshold_values = []
for score in tree_data['MasterScore']:
    if score >= 0.5:
        threshold_values.append(score)
        not_threshold_values.append(0)
    else:
        not_threshold_values.append(score)
        threshold_values.append(0)

#Greater MasterScore is = the less sustainable the city is
#Creates new MasterScore value without sustainability index (isn't relevant)
masscore_values = []
for index, row in tree_data.iterrows():
    traffic_index = row['TrafficProximityIndex']
    pop_density = row['PopDensityIndex']
    air_quality = row['AirQualityIndex']
    low_income = row['LowIncomeIndex']
    heat_imperv = row['HeatImpervIndex']
    canopy_cover = row['CanopyCoverIndex']
    new_master_score = (traffic_index + pop_density + air_quality + low_income + heat_imperv + canopy_cover)/6
    masscore_values.append(new_master_score)
tree_data['NewMasterScore'] = masscore_values

#Creates new column for what lack of canopy cover has to be for MasterScore = 0.25
thresholds = []
for index, row in tree_data.iterrows():
    traffic_index = row['TrafficProximityIndex']
    pop_density = row['PopDensityIndex']
    air_quality = row['AirQualityIndex']
    low_income = row['LowIncomeIndex']
    heat_imperv = row['HeatImpervIndex']
    new_master_score = row['NewMasterScore']
    canopy_differences = 1.5 - traffic_index - pop_density - air_quality - low_income - heat_imperv
    thresholds.append(canopy_differences)
tree_data['CanopyCoverDifference'] = thresholds #threshold - all the indices = canopy dif (- = bad)

tree_data.to_csv('updated_tree_data.csv', index=False)
#files.download('updated_tree_data.csv')
model_data = pd.read_csv('updated_tree_data.csv')

model_data = model_data.dropna(subset=['TrafficProximityIndex', 'AirQualityIndex', 'LowIncomeIndex', 'CanopyCoverDifference'])
model_data.to_csv('final_tree_data.csv', index=False)

X = model_data[['TrafficProximityIndex', 'AirQualityIndex', 'LowIncomeIndex']]
y = model_data['CanopyCoverDifference']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

!pip install -q gradio

def predict(TrafficProximityIndex, AirQualityIndex, LowIncomeIndex):
  input = pd.DataFrame([{
      'TrafficProximityIndex': float(TrafficProximityIndex),
      'AirQualityIndex': float(AirQualityIndex),
      'LowIncomeIndex': float(LowIncomeIndex)
      }])
  input.insert(0, 'const', 1.0)
  canopy_cover = model.predict(input)[0]
  return f'Predicted Canopy Cover Difference: {canopy_cover: .2f}'

interface = gr.Interface(
    fn = predict,
    inputs = [
        gr.Slider(minimum = 0.00, maximum = 1.00, value = 0.50, label = "Traffic Proximity Index"),
        gr.Slider(minimum = 0.00, maximum = 1.00, value = 0.50, label = "Air Quality Index"),
        gr.Slider(minimum = 0.00, maximum = 1.00, value = 0.50, label = "Low Income Index")
    ],
    outputs = gr.Textbox(label = "Model Prediction"),
    title = "Canopy Need Predictor",
    description = "Adjust the different indices to fit the city to determine a prediction for how much canopy cover is needed to make the city more sustainable."
)

interface.launch(share = True)
