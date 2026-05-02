import re 
from step1 import ProsesData
import string
import nltk

#dowload nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class Preprocessing(ProsesData):
    def __init__(self, base_path):
        super().__init__(base_path)

    def hapus_url(self, text: str):
        return re.sub(r'http\S+|www\.\S+', '', str(text))

    def hapus_tag(self, text: str):
        return re.sub(r'<[^>]+>', '', str(text))
        
    def hapus_mention_dan_hashtag(self, text: str):
        text = re.sub(r'@\w+', '', str(text))
        text = re.sub(r'#\w+', '', str(text))
        return text
    def hapus_tanda_baca(self,teks:str):
        return teks.translate(str.maketrans('', '', string.punctuation))
    def hapus_angka(self,teks:str):
        return re.sub(r'\d+', '', teks)

    def hapus_emoji(self,teks:str):
        return re.sub(r'[^\x00-\x7F]+', '', teks)
    
    def jalankan_cleaning(self, text: str):
        """Fungsi praktis untuk menjalankan semua proses cleaning sekaligus"""
        text = self.hapus_url(text)
        text = self.hapus_tag(text)
        text = self.hapus_mention_dan_hashtag(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.hapus_tanda_baca(text)
        text = self.hapus_angka(text)
        text = self.hapus_emoji(text)
        return text