import tensorflow as tf
from pathlib import Path
from CNN_Classifier.entity.config_entity import ModelTrainConfig

class ModelTrain:
    def __init__(self, config: ModelTrainConfig):
        self.config = config
        self.model = None

    def get_model_train(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.config.features+1,32))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh')))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))

        self.model.add(tf.keras.layers.Dense(6, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return self.model
    
    def train_model(self, train_dataset, val_dataset):
        if self.model is None:
            self.get_model_train()
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
        )
        
        return history

    def save_model(self):
        if self.model is not None:
            self.model.save(self.config.trained_model_path)
        else:
            raise ValueError("Model hasn't been trained yet. Call train_model() before saving.")

    @classmethod
    def load_model(cls, path: Path):
        return tf.keras.models.load_model(path)

