#%%
# 1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
"""
-----------------------------------TRAINING DATASET---------------------------------------------------
"""
# 2. Load data - training dataset
df = pd.read_csv('customer_segmentation.csv')

#%%
# 3. Inspect dataset
df.head()

#%%
#4. Data Cleaning - New
df.drop(columns=["ID"], inplace=True)

#%%
# check for missing values
df.isnull().sum()

#%%
# Handling Missing Values
df['Ever_Married'].fillna('Unknown', inplace=True)
df['Graduated'].fillna('Unknown', inplace=True)
df['Profession'].fillna('Unknown', inplace=True)
df['Work_Experience'].fillna(0, inplace=True)  # Assuming 0 for missing work experience
df['Family_Size'].fillna(df['Family_Size'].median(), inplace=True)
df['Var_1'].fillna('Unknown', inplace=True)

#%%
# Standardize and Clean Text (remove leading/trailing spaces)
df['Gender'] = df['Gender'].str.strip()
df['Ever_Married'] = df['Ever_Married'].str.strip()
df['Graduated'] = df['Graduated'].str.strip()
df['Profession'] = df['Profession'].str.strip()
df['Spending_Score'] = df['Spending_Score'].str.strip()
df['Var_1'] = df['Var_1'].str.strip()
df['Segmentation'] = df['Segmentation'].str.strip()

#%%
# Removing Duplicates
df.drop_duplicates(inplace=True)

#%%
# Print info after cleaning
df.info()

# %%
df.head()

# %%
df.isnull().sum()

# %%
from sklearn.preprocessing import LabelEncoder

# Encoding Categorical Variables
label_encoder = LabelEncoder()
categorical_cols = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# %%
#5. Data Splitting
X = df.drop(columns=["Segmentation"])
y = df["Segmentation"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
y_train_encoded = label_encoder.fit_transform(y_train)

# %%
#6. Train model 

# Define the model architecture
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
# Implement learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.1
    return lr

learning_rate_scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)

# %%
# Create a TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

# Add an EarlyStopping callback to prevent overfitting
early_stopping_callback = keras.callbacks.EarlyStopping(patience=5)

# Train the model
history = model.fit(X_train, y_train_encoded, epochs=20, batch_size=32, validation_split=0.1,
                    callbacks=[tensorboard_callback, early_stopping_callback, learning_rate_scheduler])

# %%
# Evaluate the model on the test set
from sklearn.metrics import f1_score
test_loss, test_accuracy = model.evaluate(X_test, label_encoder.transform(y_test)) 
print(f'Test Accuracy: {test_accuracy:.4f}')

# Calculate F1 score 

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
f1 = f1_score(label_encoder.transform(y_test), y_pred_labels, average='weighted')
print(f'F1 Score: {f1:.4f}')

#%%
# Train and evaluate a simple Machine Learning model 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate a simple Machine Learning model (Logistic Regression)
ml_model = LogisticRegression(max_iter=1000)
ml_model.fit(X_train_scaled, y_train)
y_pred_ml = ml_model.predict(X_test_scaled)
accuracy_ml = accuracy_score(y_test, y_pred_ml)
f1_ml = f1_score(y_test, y_pred_ml, average='weighted')

print(f'Accuracy (Machine Learning): {accuracy_ml:.4f}')
print(f'F1 Score (Machine Learning): {f1_ml:.4f}')

#%%
from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f'Accuracy (Random Forest): {accuracy_rf:.4f}')
print(f'F1 Score (Random Forest): {f1_rf:.4f}')

#%%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the hyperparameter distributions
param_distributions = {
    'n_estimators': randint(50, 150),  # Randomly choose between 50 to 150
    'max_depth': [None, 10, 20],  # Choose from these specific values
    'min_samples_split': randint(2, 10),  # Randomly choose between 2 to 10
    'min_samples_leaf': randint(1, 4)  # Randomly choose between 1 to 4
}

rf_model = RandomForestClassifier(random_state=42)

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=10,  
    cv=5,
    n_jobs=-1
)

# Perform RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Get the best model and its hyperparameters
best_rf_model = random_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f'Best Hyperparameters: {random_search.best_params_}')
print(f'Accuracy (Random Forest): {accuracy_rf:.4f}')
print(f'F1 Score (Random Forest): {f1_rf:.4f}')


#%%
# Model Comparison and Conclusion
if accuracy_ml > 0.60:
    print("The Logistic Regression model achieves an accuracy greater than 60%.")
    if accuracy_rf > 0.65:
        print("The Random Forest model achieves an accuracy greater than 65%.")
        print("Considering the performance, it appears that the Random Forest model performs better on this dataset.")
    else:
        print("The Random Forest model achieves an accuracy less than 65%.")
        print("Considering the performance, it appears that the Logistic Regression model is a competitive choice for this dataset.")
else:
    print("The Logistic Regression model achieves an accuracy less than 60%.")
    print("Further analysis and optimization are required to improve the model's performance.")


#%% 
import joblib

# Save the trained model in .h5 format
model.save('trained_model.h5')

# Save the label encoder in .pkl file format
joblib.dump(label_encoder, 'label_encoder.pkl')

# %%
# Save the model architecture plot as 'model_architecture_new.png'
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_architecture.png', show_shapes=True)

# Plot the model architecture using matplotlib
import matplotlib.pyplot as plt
import os

image_path = 'model_architecture.png'
if os.path.exists(image_path):
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.show()
else:
    print(f"Error: File '{image_path}' not found.")
