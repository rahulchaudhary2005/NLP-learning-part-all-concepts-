from nltk.tokenize import word_tokenize, sent_tokenize

var="Hello, how are you doing today? I hope everything is fine."
sen=sent_tokenize(var)
print(sen)
word=word_tokenize(var)
print(word)

print(f"the lowercase of the string is: {var.lower()}")