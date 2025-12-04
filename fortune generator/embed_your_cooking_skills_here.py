from keras import Sequential, layers, optimizers
import numpy as np
from keras.callbacks import EarlyStopping
import tensorflow as tf

with open("fortunes_10000.txt", "r") as f:
    text = f.read().lower()
    print(text)

#pre processing the data
chars = sorted(set(text))
print(chars)

index_to_char = dict((c,i) for c,i in enumerate(chars))
char_to_index = dict((i,c) for c, i in enumerate(chars))
print(index_to_char, char_to_index)

#now need to create the vectors to feed into the NN
#i wanna use an LSTM first -- how do i go about this?

#same thing -- sequence length, and then like stride. create the one-hot vectors and feed into an NN.

SEQ_LENGTH = 40
STRIDE = 3

sentences = []
next_letter = []
print()
for i in range(0, len(text) - SEQ_LENGTH, STRIDE):
    sentences.append(text[i: i + SEQ_LENGTH]) #i = 2: 2 to 42 (so 2 to 41, which is 40)
    next_letter.append(text[i + SEQ_LENGTH])
#now with vector embeddings
#for vector embeddings watch this vidoe https://www.youtube.com/watch?v=viZrOnJclY0
x = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(sentences),), dtype=np.int32)

#now need to fill up x and y
for i in range(len(sentences)):
    for j in range(SEQ_LENGTH):
        #the 5th sentence
        x[i][j] = char_to_index[sentences[i][j]]

    y[i] = char_to_index[next_letter[i]]

early_stop = EarlyStopping(
    monitor="val_loss",       # metric to watch
    patience=3,               # how many epochs to wait before stopping
    restore_best_weights=True # go back to the best epoch
)

model = Sequential()
model.add(layers.Embedding(input_dim=len(chars), output_dim=64, input_length=SEQ_LENGTH))  #64 dimesnions -> captures multiple overlapping uses of the character at once
#captures things like "a can be silent. a is a vowel, a can start a word, a is not punctuation" -> this is captured in 64 dimensions
#but why 64? general practice seems like -> 32 to 128 is general for character level
model.add(layers.LSTM(128))     #always need to specify input shape somewhere
model.add(layers.Dropout(0.3))
model.add(layers.Dense(len(chars), activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x, y, validation_split=0.1, batch_size=128, epochs=50, callbacks=[early_stop])

model.save("embedded_fortune.keras")

#to run this next part, comment out from "watch" to model.save

model = tf.keras.models.load_model("embedded_fortune.keras")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    #take the logits
    preds = np.log(preds) / temperature

    #this is softmax
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    prob = np.random.multinomial(1, preds, 1)
    return np.argmax(prob)

def generate(length, temperature):
    #how do i generate text? start with something
    char = np.random.randint(0, len(text) - SEQ_LENGTH)
    generated = text[char: char + SEQ_LENGTH]

    while(len(generated) < length):
        x = np.zeros((1, SEQ_LENGTH), dtype='int32')
        for j in range(SEQ_LENGTH):
            x[0][j] = char_to_index[generated[-(SEQ_LENGTH - j)]]

        predictions = model.predict(x, verbose=0)[0]
        next_char = sample(predictions, temperature)
        generated += index_to_char[next_char]

    return generated

print('----------0.2-----------')
print(generate(300, 0.2))
print('----------0.4-----------')
print(generate(300, 0.4))
print('----------0.6-----------')
print(generate(300, 0.6))
print('----------0.8-----------')
print(generate(300, 0.8))
print('----------1.0-----------')
print(generate(300, 1.0))





