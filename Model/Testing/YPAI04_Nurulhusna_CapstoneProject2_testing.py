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
-----------------------------------TESTING DATASET---------------------------------------------------
"""
# 2. Load data - training dataset
# Load the new dataset
new_df = pd.read_csv('new_customers.csv')

#%%
# 3. Inspect dataset
new_df.head()

#%%
#4. Data Cleaning - New
# check for missing values
new_df.isnull().sum()

#%%
# Handling Missing Values
# Data Cleaning
new_df.drop(columns=["ID"], inplace=True)
new_df['Ever_Married'].fillna('Unknown', inplace=True)
new_df['Graduated'].fillna('Unknown', inplace=True)
new_df['Profession'].fillna('Unknown', inplace=True)
new_df['Work_Experience'].fillna(0, inplace=True)
new_df['Family_Size'].fillna(new_df['Family_Size'].median(), inplace=True)
new_df['Var_1'].fillna('Unknown', inplace=True)
new_df['Gender'] = new_df['Gender'].str.strip()
new_df['Ever_Married'] = new_df['Ever_Married'].str.strip()
new_df['Graduated'] = new_df['Graduated'].str.strip()
new_df['Profession'] = new_df['Profession'].str.strip()
new_df['Spending_Score'] = new_df['Spending_Score'].str.strip()
new_df['Var_1'] = new_df['Var_1'].str.strip()
new_df.drop_duplicates(inplace=True)


#%%
# Removing Duplicates
new_df.drop_duplicates(inplace=True)

#%%
# Print info after cleaning
new_df.info()

# %%
new_df.head()

# %%
new_df.isnull().sum()

# %%
from sklearn.preprocessing import OneHotEncoder

# Encoding Categorical Variables
onehot_encoder = OneHotEncoder()
categorical_cols = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
# Encoding Categorical Variables
for col in categorical_cols:
    new_df[col] = label_encoder.fit_transform(new_df[col])

#%%
# Check for missing values in the target variable
y_new.isnull().sum()

# Handle missing values in the target variable
y_new.fillna('Unknown', inplace=True) 

# %%
#5. Data Splitting
X_new = new_df.drop(columns=["Segmentation"])  # Use new_df here
y_new = new_df["Segmentation"]  # Use new_df here
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y_new_train) 

#%%
from keras.utils import to_categorical

y_new_train_encoded = to_categorical(label_encoder.fit_transform(y_new_train), num_classes=4)
y_new_test_encoded = to_categorical(label_encoder.transform(y_new_test), num_classes=4)

# %%
#6. Train model 

# Define the model architecture
model_new = keras.models.Sequential()
model_new.add(keras.layers.Dense(128, activation='relu', input_shape=(X_new_train.shape[1],)))
model_new.add(keras.layers.BatchNormalization())
model_new.add(keras.layers.Dropout(0.3))
model_new.add(keras.layers.Dense(64, activation='relu'))
model_new.add(keras.layers.BatchNormalization())
model_new.add(keras.layers.Dropout(0.3))
model_new.add(keras.layers.Dense(32, activation='relu'))
model_new.add(keras.layers.BatchNormalization())
model_new.add(keras.layers.Dropout(0.3))
model_new.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
optimizer_new = keras.optimizers.Adam(learning_rate=0.001)
model_new.compile(optimizer=optimizer_new, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
# Implement learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.1
    return lr

learning_rate_scheduler_new = keras.callbacks.LearningRateScheduler(lr_scheduler)

# %%
# Create a TensorBoard callback
tensorboard_callback_new = keras.callbacks.TensorBoard(log_dir='./logsNew', histogram_freq=1)

# Add an EarlyStopping callback to prevent overfitting
early_stopping_callback_new = keras.callbacks.EarlyStopping(patience=3)

# Train the model
history_new = model_new.fit(X_new_train, y_new_train_encoded, epochs=20, batch_size=32, validation_split=0.1,
                    callbacks=[tensorboard_callback_new, early_stopping_callback_new, learning_rate_scheduler_new])

# %%
# Evaluate the model on the test set
test_loss_new, test_accuracy_new = model_new.evaluate(X_new_test, y_new_test_encoded ) 
print(f'Test Accuracy: {test_accuracy_new:.4f}')

# Calculate F1 score (you may need to import it from sklearn)
from sklearn.metrics import f1_score
y_pred_labels_new = np.argmax(y_pred_prob_new, axis=1)
y_true_labels_new = np.argmax(y_new_test_encoded, axis=1)
f1_new = f1_score(y_true_labels_new, y_pred_labels_new, average='weighted')
print(f'F1 Score: {f1_new:.4f}')

#%%
# Get the predicted probabilities for each class
y_pred_prob_new = model_new.predict(X_new_test)

# Convert the predicted probabilities to labels (A, B, C, D)
predicted_segmentation_encoded = np.argmax(y_pred_prob_new, axis=1)
predicted_segmentation = label_encoder.inverse_transform(predicted_segmentation_encoded)
print(predicted_segmentation)

#%%
import joblib

# Save the trained model in .h5 format
model_new.save('trained_model_test.h5')

# Save the label encoder in .pkl file format
joblib.dump(label_encoder, 'label_encoder_test.pkl')

# %%

# Save the model architecture plot as 'model_architecture_new.png'
from tensorflow.keras.utils import plot_model

plot_model(model_new, to_file='model_architecture_new.png', show_shapes=True)

# Plot the model architecture using matplotlib
import matplotlib.pyplot as plt
import os

image_path = 'model_architecture_new.png'
if os.path.exists(image_path):
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.show()
else:
    print(f"Error: File '{image_path}' not found.")


# %%
