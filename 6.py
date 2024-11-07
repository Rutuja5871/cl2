import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

documents = {
    1: "Natural language processing and information retrieval are key components of modern search engines.",
    2: "Inverted indexing improves the efficiency of search by storing occurrences of terms.",
    3: "Stemming and stop word removal can be applied to terms in an inverted index.",
    4: "Information retrieval systems retrieve relevant documents based on a search query.",
}

def preprocess_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return stemmed_words

def create_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, text in documents.items():
        words = preprocess_text(text)
        for word in words:
            if doc_id not in inverted_index[word]:
                inverted_index[word].append(doc_id)
    return dict(inverted_index)

def search(query, inverted_index):
    query_terms = preprocess_text(query)
    doc_sets = [set(inverted_index.get(term, [])) for term in query_terms]
    if doc_sets:
        result_docs = set.intersection(*doc_sets)
    else:
        result_docs = set()
    return result_docs

inverted_index = create_inverted_index(documents)
print("Inverted Index:", inverted_index)

query = "information retrieval"
result_docs = search(query, inverted_index)
print("\nDocuments matching query '{}': {}".format(query, result_docs))
