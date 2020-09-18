This is a simple project with the intention to classify whether a tweet is POSITIVE, NEGATIVE or NEUTRAL. 
The first step was to clean the data and since this is a text based data, I had to use NLP for this purpose. I used porterstemmer since it is the least aggressive of all 
stemming. The data is only in english so to remove common words (like a, the, that etc.) I downloaded stopwords from nltk package and scrubbed these words and created my corpus.
After this is used TfidfVectorizer to calculate Term Frequency - Inverse Document Frequency of the words, this would help the model to put emphasis only on the words which
have significance in a sentence.
The next step is to apply a learning algorithm, which is the Naive Bayes classifier in this case.
