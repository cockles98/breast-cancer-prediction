import tensorflow as tf
import pandas as pd

# Load the model
loaded_model = tf.keras.models.load_model('D:\\GitHub Dic\\breast_cancer_prediction\\saved_model')

# Print model summary
loaded_model.summary()

# Compile the model (if necessary)
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load test data
test_data = pd.read_csv('D:\\GitHub Dic\\breast_cancer_prediction\\dataset\\labels_dataset.csv')
test_labels = pd.read_csv('D:\\GitHub Dic\\breast_cancer_prediction\\dataset\\target_dataset.csv')

# Evaluate the model
loss, accuracy = loaded_model.evaluate(test_data, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
