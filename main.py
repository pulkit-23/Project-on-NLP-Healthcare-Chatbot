import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

terms=[]
tags = []
doc = []
ignwd = ['?', '!']
df = open('intents.json').read()
intents = json.loads(df)

#print(intents)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        terms.extend(w)
        doc.append((w, intent['tag']))

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

#print(terms)
#print(doc)-each pattern tokenized, connected to thier tags
#print(tags)-show all tags

terms = [lemmatizer.lemmatize(w.lower()) for w in terms if w not in ignwd]

#print(terms)-convert to lower case and remove ignwd

terms = sorted(list(set(terms)))
#print(terms)-sorts alphabetically

tags = sorted(list(set(tags)))

#print(tags)- print all tags alphabetically

#print (len(doc), "doc")-231

#print (len(tags), "tags", tags)-47

#print (len(terms), "unique lemmatized terms", terms)-242

pickle.dump(terms,open('terms.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))

training = []
outemp = [0] * len(tags)

#print(outemp)

for d in doc:
    collec = []
    patw = d[0]
    patw = [lemmatizer.lemmatize(word.lower()) for word in patw]
    for w in terms:
        collec.append(1) if w in patw else collec.append(0)

    outrow = list(outemp)
    outrow[tags.index(d[1])] = 1

    training.append([collec, outrow])
    
#print(patw)-traversing by storing each pattern
#print(collec)-store 1 for each pattern word and 0 for other
#print(outrow)-store 1 corresponding to the tag of that pattern 
#print(training)

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

#print(train_x)
#print(train_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")







