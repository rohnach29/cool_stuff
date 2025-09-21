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
for i in range(0, len(text) - SEQ_LENGTH, STRIDE):
    sentences.append(text[i: i + SEQ_LENGTH]) #i = 2: 2 to 42 (so 2 to 41, which is 40)
    next_letter.append(text[i + SEQ_LENGTH])
#let's try with one-hot vectors first
# x = np.zeros((len(sentences), SEQ_LENGTH, len(chars)), dtype=np.float32)
# y = np.zeros((len(sentences), len(chars)), dtype=np.float32)

# #now need to fill up x and y
# for i in range(len(sentences)):
#     for j in range(SEQ_LENGTH):
#         x[i][j][char_to_index[sentences[i][j]]] = 1

#     y[i][char_to_index[next_letter[i]]] = 1

# early_stop = EarlyStopping(
#     monitor="val_loss",       # metric to watch
#     patience=3,               # how many epochs to wait before stopping
#     restore_best_weights=True # go back to the best epoch
# )

# model = Sequential()
# model.add(layers.LSTM(128, input_shape=(SEQ_LENGTH, len(chars))))       #always need to specify input shape somewhere. input_shape is (time dimensions, feature dimension)
# model.add(layers.Dropout(0.3))
# model.add(layers.Dense(len(chars), activation="softmax"))

# model.compile(optimizer="adam", loss="categorical_crossentropy")
# model.fit(x, y, validation_split=0.1, batch_size=128, epochs=50, callbacks=[early_stop])

# model.save("fortune.keras")

model = tf.keras.models.load_model("fortune.keras")

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
    char = np.random.randint(0, len(text))
    generated = text[char: char + SEQ_LENGTH]

    while(len(generated) < length):
        x = np.zeros((1, SEQ_LENGTH, len(chars)), dtype='float32')
        for j in range(SEQ_LENGTH):
            x[0][j][char_to_index[generated[-(SEQ_LENGTH - j)]]] = 1

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



