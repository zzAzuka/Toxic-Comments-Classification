import tensorflow as tf
from pathlib import Path
from CNN_Classifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(input_dim=self.config.vocab_size,
                                                  output_dim=self.config.embedding_dim,
                                                  input_length=self.config.max_sequence_length))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.config.lstm_units1, return_sequences=True)))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.config.lstm_units2)))
        self.model.add(tf.keras.layers.Dense(self.config.dense_units1, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(self.config.dropout_rate1))
        self.model.add(tf.keras.layers.Dense(self.config.dense_units2, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(self.config.dropout_rate2))
        self.model.add(tf.keras.layers.Dense(self.config.classes, activation='sigmoid'))

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

