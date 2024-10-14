import os
import urllib.request as request
import zipfile
from CNN_Classifier import logger
from CNN_Classifier.utils.common import get_size
from CNN_Classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import pandas as pd
from tensorflow.keras.layers import TextVectorization # type: ignore
import tensorflow as tf

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def load_data(self):
        csv_files = [f for f in os.listdir(self.config.unzip_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the extracted directory")
        
        csv_path = os.path.join(self.config.unzip_dir, csv_files[0])
        
        # Load and preprocess CSV
        df = pd.read_csv(csv_path)
        logger.info(f"CSV file {csv_path} loaded successfully.")
        
        X = df['comment_text'].dropna()
        y = df[df.columns[2:]].dropna().values
        return X,y

    def vectorization_dataset(self,X,y):
        MAX_FEATURES = self.config.features
        vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=self.config.output_len, output_mode='int')
        vectorizer.adapt(X.values)
        vectorized_text = vectorizer(X.values)
        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.cache()
        dataset = dataset.shuffle(160000)
        dataset = dataset.batch(16)
        dataset = dataset.prefetch(8)
        train = dataset.take(int(len(dataset)*.8))
        val = dataset.skip(int(len(dataset)*.8)).take(int(len(dataset)*.1))
        test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

        return train, val, test
 
    


    