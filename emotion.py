import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import string
import numpy as np
import random

nltk.download('stopwords')

data = {
    'Text': [
        "I feel great today!",
        "I am so sad and lonely.",
        "This is so frustrating!",
        "What an amazing experience!",
        "I'm scared of what will happen.",
        "I'm so angry right now!",
        "I'm ecstatic about my new job!",
        "I can't take all this stress anymore.",
        "This is a wonderful surprise!",
        "I am so angry with you!",
        "I feel happy when I am with my friends.",
        "This movie made me feel so sad.",
        "I can't believe how scared I am!",
        "Such a wonderful day!",
        "Why does everything go wrong? It's so frustrating!",
        "I just want to sleep, everything is so overwhelming.",
        "I love spending time with my family.",
        "I feel so relaxed in nature.",
        "I am terrified of this new journey.",
        "Everything is going so well, I feel on top of the world!",
        "I can never get things right, everything frustrates me."
    ],
    'Emotion': ['joy', 'sadness', 'anger', 'joy', 'fear', 'anger', 'joy', 'fear', 'joy', 'anger', 'joy', 'sadness', 'fear', 'joy', 'anger', 'joy', 'fear', 'joy', 'anger']
}

df = pd.DataFrame(data)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Text'] = df['Text'].apply(preprocess_text)

X = df['Text']
y = df['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_emotions(texts):
    processed_texts = [preprocess_text(text) for text in texts]
    tfidf_texts = vectorizer.transform(processed_texts)
    predictions = model.predict(tfidf_texts)
    return predictions

sample_texts = [
    "I am feeling on top of the world!",
    "I can't handle all this pressure.",
    "I'm so excited for tomorrow!",
    "Why is everything so overwhelming?",
    "I love spending time with my friends, it makes me feel so joyful!",
    "I'm absolutely terrified of what's coming next.",
    "It feels so good to have achieved everything I wanted!",
    "Why does everything feel so hopeless right now?",
    "I feel so peaceful when I meditate.",
    "What a stressful week, I can't take it anymore!"
]

predictions = predict_emotions(sample_texts)

for text, prediction in zip(sample_texts, predictions):
    print(f"Text: {text}\nPredicted Emotion: {prediction}\n")

def generate_random_text(num_samples):
    emotions = ['joy', 'sadness', 'anger', 'fear']
    sentences = [
        "I am feeling so happy today!",
        "Everything is falling apart, I feel terrible.",
        "This is the worst moment of my life.",
        "I'm in a state of pure bliss.",
        "What is happening, I'm scared.",
        "This situation is driving me mad.",
        "I feel elated like never before.",
        "I feel like I am losing control.",
        "This moment makes me feel alive.",
        "I don't know how I feel, it's overwhelming."
    ]
    random_texts = []
    random_labels = []
    
    for _ in range(num_samples):
        random_sentiment = random.choice(emotions)
        random_sentence = random.choice(sentences)
        random_texts.append(random_sentence)
        random_labels.append(random_sentiment)
    
    return random_texts, random_labels

random_texts, random_labels = generate_random_text(10)
random_predictions = predict_emotions(random_texts)

print("\nRandom Sample Predictions:")
for text, label, prediction in zip(random_texts, random_labels, random_predictions):
    print(f"Text: {text}\nTrue Emotion: {label}\nPredicted Emotion: {prediction}\n")

def evaluate_custom_input(sentences):
    processed_texts = [preprocess_text(sentence) for sentence in sentences]
    tfidf_matrix = vectorizer.transform(processed_texts)
    pred_emotions = model.predict(tfidf_matrix)
    return list(zip(sentences, pred_emotions))

custom_input = [
    "This week has been incredible, I can't stop smiling!",
    "I am utterly terrified of what lies ahead.",
    "I just want to escape this feeling of constant worry.",
    "Finally, something good is happening in my life!",
    "I feel so disconnected from everything around me."
]

evaluation_results = evaluate_custom_input(custom_input)

print("\nCustom Input Evaluations:")
for sentence, prediction in evaluation_results:
    print(f"Sentence: {sentence}\nPredicted Emotion: {prediction}\n")

complex_sentences = [
    "I'm not sure whether I should be happy or sad about this change.",
    "This is the most frustrating situation I've ever faced in my life!",
    "I feel so proud of what I've achieved, but there's still a lot to be done.",
    "Sometimes, I wish I could feel less anxious about everything."
]

complex_results = evaluate_custom_input(complex_sentences)

print("\nComplex Sentence Evaluations:")
for sentence, prediction in complex_results:
    print(f"Sentence: {sentence}\nPredicted Emotion: {prediction}\n")
