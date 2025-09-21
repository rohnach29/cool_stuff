from textblob import TextBlob
from newspaper import Article

#very high level nlp sentiment analysis
url = 'https://en.wikipedia.org/wiki/Mathematics'
article = Article(url)

article.download()
article.parse() #get all the HTML out
article.nlp()   #prepare it for natural language processing

text = article.summary
print(text)

blob = TextBlob(text)   #Without creating the TextBlob object, you’d have to manually call lower-level NLTK functions and handle preprocessing yourself.

sentiment = blob.sentiment.polarity
print(sentiment)

#okay this is super simple, but let's dive into NLP and how this sentiment analysis works