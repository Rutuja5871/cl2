import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

text = "Natural language processing allows computers to understand human language. Preprocessing steps like stemming and removing stop words make this task easier."

def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word in words:
        if word.lower() not in stop_words:
            filtered_words.append(word)
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return stemmed_words

processed_text = preprocess_text(text)
print("Processed Text:", processed_text)
