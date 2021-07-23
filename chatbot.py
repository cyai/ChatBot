import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import numpy as np


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    responce = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(responce) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_lsit = []
    for r in results:
        return_lsit.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
    return return_lsit

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")
print("HELP ASSISTANCE!! \n")
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n")

while True:
    usr_message = input("YOU: ")
    usr_message = usr_message.lower()
    ints = predict_class(usr_message)
    response = get_response(ints, intents)
    print(f"BOT: {response}\n")
    response = response.lower()

    if 'exit' in response or 'goodbye' in response or 'thank you' in response or 'bye' in response:
        break


    if 'course' in response or 'skills' in response or 'learn' in response:
        course_type = input("Which type of course do you prefer to learn? \n 1) Python from scratch \n 2) SPEAK CHINESE WITH US \n 3) CODING LEARNING SCRATCH THE EASY WAY \n 4)LEARN SPANISH THE EASY WAY \n 5) View More.....\n \n* Your Response:  ")
        course_type = course_type.lower()
        if course_type == '5' or course_type == 'more' or course_type == 'view more':
            more_type = input("What do you want to learn? \n 1) Piano \n 2) Harmonium \n 3) Singing \n 4) Learn Hindi \n 5) Chess \n 6) Flute \n 7) Yoga \n 8) Vedic Mathematics \n 9) Indian Tabla \n 10) Violin \n 11) Sitar \n 12) Guitar \n 13) Personality Development \n 14) Spanish \n 15) Chinese \n 16) Python Programming \n 17) Computer Coading \n \n* Your Response: ")
            if more_type:
                print("YET TO CODE]\n")

            continue
        else:
            print("YET TO CODE\n")
    
    if 'executive' in response or 'connect' in response or 'call' in response:
        permission = input("Yes/No(Y/n): ")
        permission = permission.lower()
        if permission == 'yes' or permission == 'y':
            print("Connecting......\n Please wait..")
        elif permission == 'no' or permission == 'n':
            more_help = input("Do you need any more help? \n")
            more_help = more_help.lower()
            
            if more_help == 'yes' or more_help == 'y' or more_help == 'sure' or more_help == 'think so':
                print("Go ahead I'm listning...")
                continue
            else:
                print("Thank You! Bye")
                break
        