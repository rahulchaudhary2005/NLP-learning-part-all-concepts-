import spacy

nlp=spacy.blank("en")

doc=nlp("./data.txt")
for token in doc:
    print(token)
