import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import pickle
import json
from memorynetwork import tokenize  # Assuming tokenize is in memorynetwork.py

def predict_answer(story, question, model, word_idx, story_maxlen, query_maxlen):
    story_tokens = []
    for sentence in story:
        story_tokens.extend(tokenize(sentence))
    question_tokens = tokenize(question)
    story_seq = [word_idx.get(word, 0) for word in story_tokens]
    question_seq = [word_idx.get(word, 0) for word in question_tokens]
    story_padded = pad_sequences([story_seq], maxlen=story_maxlen)[0]
    question_padded = pad_sequences([question_seq], maxlen=query_maxlen)[0]
    prediction = model.predict([story_padded, question_padded])[0]
    answer_idx = np.argmax(prediction)
    if answer_idx == 0:
        return "Unknown word"
    else:
        for word, idx in word_idx.items():
            if idx == answer_idx:
                return word
        return "No answer found"

model = load_model('memory_network_model.h5')
with open('word_idx.pickle', 'rb') as f:
    word_idx = pickle.load(f)
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
story_maxlen = metadata['story_maxlen']
query_maxlen = metadata['query_maxlen']

# Example usage
story = ["Mary moved to the bedroom.", "John went to the kitchen."]
question = "Where is Mary?"
answer = predict_answer(story, question, model, word_idx, story_maxlen, query_maxlen)
print(answer)