from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
# import spacy

# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("emoji", first=True)

#model and tokenizer
roberta  = 'cardiffnlp/twitter-roberta-base-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['negative', 'neutral', 'positive']

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# sentiment_analysis = pipeline("sentiment-analysis", model=model_name, top_k = None)
def sentiment_with_emotion(tweet):

    tweet_processed = tweet

    #inference
    encoded_tweet = tokenizer(tweet_processed, return_tensors='pt')

    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    #results
    result = {labels[i]: float(scores[i]) for i in range(3)}
    return scores[0]


# def perform_sentiment_analysis(text):
#   results = sentiment_analysis(text)
#   return results[0][0]['score']


# classifier = pipeline("text-classification",model='HannahRoseKirk/Hatemoji', top_k=None)
def analyse(text):
    # doc = nlp(text)
    # if doc._.has_emoji:
    #     return sentiment_with_emotion(text)
    # else:
    #     return perform_sentiment_analysis(text)
    return sentiment_with_emotion(text)
if __name__ == "__main__":
   text = 'I am sad ❤️'
   print(sentiment_with_emotion(text))