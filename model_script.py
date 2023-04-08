import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
data = pd.read_csv(r'C:\Users\hanna\PycharmProjects\COMP-4949-Big-Data\A2\gender_classification_v7.csv')

# Select the features we want to use
selected_features = [
    "long_hair",
    "forehead_width_cm",
    "forehead_height_cm",
    "nose_wide",
    "nose_long",
]
X = data[selected_features]
y = data["gender"]

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
