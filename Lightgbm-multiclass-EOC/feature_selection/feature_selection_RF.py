import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# Reading CSV file
data = pd.read_csv("./dataset_class2/2d_LVD_radiomic.csv")

# Delete first column
data = data.drop(data.columns[0], axis=1)
X = data.drop('TSR', axis=1)
y = data['TSR']

# Separate features from labels
scaler = StandardScaler()

# The training set feature data is normalized by Z-score and the feature names are retained
X_train_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Create a random forest model object
np.random.seed(0)
rf_model = RandomForestClassifier(n_estimators=100)

# Training
rf_model.fit(X_train_scaled, y)

# Create a TreeExplainer
explainer = shap.TreeExplainer(rf_model)

# Gets the SHAP value of binary class
shap_values = explainer.shap_values(X_train_scaled)
max_display = 25

class_names = ["low", "high"]

# Generate a Summary Plot that shows the effect of the SHAP value for each feature on the model output
plt.figure(figsize=(10, 6))
summary_plot = shap.summary_plot(shap_values, X_train_scaled, max_display=max_display,
                                 class_names=class_names, show=False)

plt.savefig("./output/2d_LVD_radiomic_summary_plot.pdf", format='pdf')
plt.show()

# Draw SHAP diagrams
class_idx = 0
shap.summary_plot(shap_values[class_idx], X_train_scaled, class_names=['low'], max_display=max_display, show=False)

plt.savefig("./output/LVD_radiomic_low.pdf", format='pdf')
plt.show()

class_idx = 1
shap.summary_plot(shap_values[class_idx], X_train_scaled, class_names=['high'],
                  max_display=max_display, show=False)

plt.savefig("./output/LVD_radiomic_high.pdf", format='pdf')
plt.show()
