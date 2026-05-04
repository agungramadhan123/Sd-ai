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


    def hapus_emoji(self,teks:str):
        return re.sub(r'[^\x00-\x7F]+', '', teks)
    
    def hapus_placeholder(self, text: str):
        text = re.sub(r'\{@[^}]*@\}', '', str(text))
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        return text

    def jalankan_cleaning(self, text: str):
        text = str(text).lower()
        text = self.hapus_url(text)
        text = self.hapus_tag(text)
        text = self.hapus_mention_dan_hashtag(text)
        text = self.hapus_placeholder(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.hapus_tanda_baca(text)
        text = self.hapus_emoji(text)
        return text

    def tokenize_dan_hapus_stopwords(self, text: str):
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words('english'))
        custom_stopwords = {
            'username', 'url', 'via',
            'rt', 'amp',
        }
        stop_words.update(custom_stopwords)

        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return ' '.join(tokens)
    
    def preprocess_full(self, text: str):
        text = self.jalankan_cleaning(text)
        text = self.tokenize_dan_hapus_stopwords(text)
        return text