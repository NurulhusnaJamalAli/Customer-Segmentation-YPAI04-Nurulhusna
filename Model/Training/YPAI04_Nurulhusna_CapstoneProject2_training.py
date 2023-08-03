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
test_loss, test_accuracy = model.evaluate(X_test, label_encoder.transform(y_test)) 
print(f'Test Accuracy: {test_accuracy:.4f}')

# Calculate F1 score (you may need to import it from sklearn)
from sklearn.metrics import f1_score
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
f1 = f1_score(label_encoder.transform(y_test), y_pred_labels, average='weighted')
print(f'F1 Score: {f1:.4f}')

#%% 
import joblib

# Save the trained model in .h5 format
model.save('trained_model.h5')

# Save the label encoder in .pkl file format
joblib.dump(label_encoder, 'label_encoder.pkl')

# %%
