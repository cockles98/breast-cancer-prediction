import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

class MyClassifier(tf.keras.Model):
    def __init__(self, optimizer, loss, kernel_initializer, neurons):
        super(MyClassifier, self).__init__()
        self.dense1 = Dense(units=neurons, activation='relu',
                            kernel_initializer=kernel_initializer,
                            input_dim=30)
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(units=neurons, activation='relu',
                            kernel_initializer=kernel_initializer)
        self.dropout2 = Dropout(0.2)
        self.output_layer = Dense(units=1, activation='sigmoid')
        self.optimizer = optimizer
        self.loss = loss
        self.metric = ['accuracy']

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)
    
# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()

# Define kernel initializer and number of neurons
kernel_initializer = 'random_uniform'
neurons = 32

# Instantiate the model
model = MyClassifier(optimizer, loss, kernel_initializer, neurons)

# Pass some dummy input shape to build the model
model.build(input_shape=(None, 30))

# Compile the model
model.compile(optimizer=model.optimizer,
              loss=model.loss,
              metrics=model.metrics)

# Load data
labels = pd.read_csv('D:\\GitHub Dic\\breast_cancer_prediction\\dataset\\features.csv')     # Put the current location of features.csv file
target = pd.read_csv('D:\\GitHub Dic\\breast_cancer_prediction\\dataset\\classes.csv')      # Put the current location of classes.csv file

# Train the model
model.fit(labels, target, epochs=1200, batch_size=32, validation_split=0.2)

# Save the model
model.save('saved_model')
