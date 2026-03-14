from turtle import pu

from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
var="This is a sample sentence, showing off the stop words filtration. Stop words are commonly used words that are often removed from text data during natural language processing tasks to improve the efficiency and accuracy of the analysis."

word=word_tokenize(var)
print("Original Words:", word)

stop=list(punctuation)+stopwords.words('english')

# Removing stop words and punctuation
for i in word:
    if i in stop:
        word.remove(i)
print("After Removing Stop Words and Punctuation:", word)

