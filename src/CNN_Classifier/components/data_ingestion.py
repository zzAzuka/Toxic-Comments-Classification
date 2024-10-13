import os
import urllib.request as request
import zipfile
from CNN_Classifier import logger
from CNN_Classifier.utils.common import get_size
from CNN_Classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import pandas as pd
import re


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
    
    def clean_text(self, text):
        if isinstance(text, str):
            text = text.replace('\n', ' ')
            text = re.sub(r"[^a-zA-Z0-9\s.,!?']", ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        return ""

    def load_data(self):
        csv_files = [f for f in os.listdir(self.config.unzip_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the extracted directory")
        
        csv_path = os.path.join(self.config.unzip_dir, csv_files[0])
        
        # Load and preprocess CSV
        df = pd.read_csv(csv_path)
        logger.info(f"CSV file {csv_path} loaded successfully.")
        
        df = self.preprocess_data(df)
        return df

    def preprocess_data(self, df):
        df = df.dropna()  # Example of basic preprocessing
        if 'comment_text' in df.columns:
            df['comment_text'] = df['comment_text'].apply(self.clean_text)
        return df
    


    