from operator import le

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
var = "The striped bats are hanging on their feet for best"
# Stemming
ps = PorterStemmer()
word = nltk.word_tokenize(var)
print("Original Words:", word)
stemmed_words = [ps.stem(w) for w in word]
print("Stemmed Words:", stemmed_words)
# Lemmatization
lemmatizer = WordNetLemmatizer()
new_word=lemmatizer.lemmatize("hanging", pos="v")
print(f"The lemmatized form of 'hanging' is: {new_word} ")
