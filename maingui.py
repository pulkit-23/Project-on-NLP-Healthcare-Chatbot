import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
term = pickle.load(open('terms.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))


def cleanmsg(message):
    msgwd = nltk.word_tokenize(message)
    msgwd = [lemmatizer.lemmatize(word.lower()) for word in msgwd]
    return msgwd

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def func(message, term, det=True):
    msgwd = cleanmsg(message)
    collc = [0]*len(term)
    for s in msgwd:
        for i,w in enumerate(term):
            if w == s:
                collc[i] = 1
                if det:
                    print ("found in bag: %s" % w)
    return(np.array(collc))

def predtag(message, model):
    p = func(message, term,det=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    final = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    final.sort(key=lambda x: x[1], reverse=True)
    rlist = []
    for r in final:
        rlist.append({"intent": tags[r[0]], "probability": str(r[1])})
    return rlist

def get_Res(ints, jsonint):
    tag = ints[0]['intent']
    intslist = jsonint['intents']
    for i in intslist:
        if(i['tag']== tag):
            final = random.choice(i['responses'])
            break
    return final

def botres(msg):
    ints = predtag(msg, model)
    res = get_Res(ints, intents)
    return res



import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = botres(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("HealthCare Chatbot")
base.geometry("620x700")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="orange", height="30", width="60", font="Arial",)

ChatLog.config(state=NORMAL)
ChatLog.insert(END, "Welcome to Healthcare Chatbot! Enter all your symptoms precisely."+'\n\n')
ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="17", height=7,
                    bd=0, bg="#061f8f", activebackground="#3cb0de",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="#85c4de",width="49", height="8", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=596,y=6, height=586)
ChatLog.place(x=6,y=6, height=586, width=590)
EntryBox.place(x=148, y=601, height=90, width=448)
SendButton.place(x=6, y=601, height=90)

base.mainloop()