import random
import numpy as np
import tensorflow as tf
import os
import keras
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import Sequential, layers, optimizers


filepath = keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")  #keras downloads it from the internet and stores it in its cache, so that it doesn't have to download it again next time
text = open(filepath, "rb").read().decode(encoding='utf-8').lower() #why lower? with all upper, more options -> more failure. model predicts character by character. all lower makes it more accurate

# reads the entire file as raw bytes and then converts the bytes into a string using utf-8 decoding
# why read in binary first? Sometimes files downloaded from the internet can contain byte sequences that aren’t guaranteed to match the system’s default encoding — reading as binary avoids decoding issues until you explicitly choose UTF-8.
# so the system has a default encoding that might not be utf-8, so we open it as a binary first and explicitly say decode using utf-8 because we know it's encoded in that

# ----- NEXT STEP -------
#need to convert each character into a unique numerical representation

#just going to select one part of the text for now, training on the whole things might take too long
text = text[300000:800000]

characters = sorted(set(text)) #set() filters out all the unique characters
print(characters)

index_to_char = dict((c, i) for (c,i) in enumerate(characters))  #0:a, 1:b...
char_to_index = dict((c, i) for (i,c) in enumerate(characters))  #a:0, b:1...

SEQ_LENGTH = 40 #this is the number of characters we feed into the neural net, and it then predicts the 41st character
STEP_SIZE = 3


#ORIGINAL CODE TO TRAIN MODEL:
'''
sentences = []  #input into the model
next_characters = []  #target character -> what the next character is SUPPOSED TO BE given the sentence

#for loop: goes from 0 until the last sequence (otherwise get index out of bounds), with a step size of 3
#i wonder: is this off by 1? say i have 10, seq length 5. goes from 0 to 4 inclusive. abcdefghij. sentences appends abcde, next char appends f.
#then, cdefg, next char i. then, efghi, next char j. wait, that's perfect. surely there's a case where it messes up.
#step size of 5? still works. 

#okay so there just might be some training samples at the end that get skipped, that's what i was checking.
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)  #basically, in (ex) sentence 1 at pos 34, if character 'x' occurs, this will be 1. all the other [1,34,something] will be 0
y = np.zeros((len(sentences), len(characters)), dtype=bool) #for sentence 5, the next character is char 8 -> [5,8] will be 1. [5,{1,2,3,4...}] will be 0.

#now just need to fill up the arrays
for i in range(len(sentences)):
    for j in range(SEQ_LENGTH):
        x[i, j, char_to_index[sentences[i][j]]] = 1

    y[i, char_to_index[next_characters[i]]] = 1

#data prepared! train the RNN now.
model = Sequential()
#a bit of understanding -- RNNs take the hidden state (previous output) into consideration because it's essentially a summary of everything that has happened so far
# why don't we just take all the inputs from 1-5 at timestamp 6? this would grossly increase the number of things weights we would need to remember
# RNNs have a vanishing/exploding gradient problem -- something that happened a loooong time ago would get hugely exploded/vanish because of unrolling
#if it unrolls 50 times and w = 2, the gradient can EXPLODE in backprop bc the number 2^50 shows up in some places. similarly w vanishing
# --- SOLUTION: LSTM (Long Short-Term Memory -- weird name man) ----
#there are 2 main paths here -> things to remember long term and things to remember short-term
# for long-term things, weights and biases don't influence them so there's no vanishing/exploding gradient
# https://www.youtube.com/watch?v=YCzL96nL7j0 -- watch this to understand LSTMs
# -- FORGET GATE --
# the forget gate (man i need to create a README or just a txt file for these notes) -- the first step of LSTM is the "forget gate" -- it determines what PERCENTAGE of the LSTM we need to remember
# how? first it processed the short-term memory stuff -- similar to an RNN, it takes sum of weight*input + weight2*previous + bias -> sigmoid -> this output gets multiplied w the previous long-term memory input
# if input is very negative, there is a chance that after the sigmoid, we get 0 -> remember 0% of the long term stuff

# -- INPUT GATE -- 
# has 2 parts: "potential long-term memory" and "% of potential long-term memory" -> 2nd part is calculated same as before, first part is w*input + w2*prev_output (different weights btw) + bias -> tanh function
# multiply 1,2 and add to long-term memory -> new long-term memory

# -- OUTPUT GATE --
# determines new short-term memory.
# takes new long-term memory from before -> tanh activation. then determines % of this to remember (same from before) and multiplies to gives new short-term memory! 

#for next time: still have to understand why 128 and input_shape!
#128 different LSTM neurons -- means 128 different notebooks, each with their own input, output, forget gate
#input to each is (40, length(characters)) -> it basically processes 40 timestamps, and len(characters) signifies which character is "1" at that timestamp
model.add(layers.LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
#LSTM -> fully connected later. Dense means we have len(characterse) neurons -> convert this into probabilities of next character with softmax

#need to understand LSTM -> Dense layer. full flow
model.add(layers.Dense(len(characters)))
model.add(layers.Activation('softmax'))

#what is RMSprop? optimization. learns from previous things. basically if gradient high -> lr low, gradient low 
#categorical_crossentropy -> 
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.01))
model.fit(x,y,batch_size=256, epochs=4) #why power of 2? 256? what is epochs? -> power of 2 bc easier for CPUs. epochs is number of times it goes through training data.

model.save('textgenerator.keras')
'''

model = tf.keras.models.load_model('textgenerator.keras')

#need to understand this function. basics: samples one character from model. temperature = 0 -> safe. pick higher probs. temperature == 1 -> experimental. probably won't make much sense.
# at temperate < 1, makes the model MORE confident. a =0.7 -> more likely to choose a. at temperate > 1 , flatten probabilities. makes it more creative but more prone to gibberish

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64') #just gives extra precision
    preds = np.log(preds) / temperature     #take log first because if we want to make it sharper/flatter, it's easier to work in a log space than in probabilities, which are between 0 and 1. divide by temp -> if < 1, differences are magnified. > 1: differences are flattened
    #t = 0.5: [-0.36, -1.61, -2.30] → [-0.72, -3.22, -4.60]. index 0 is much less negative and dominates when we do softmax again

    #softmax: convert back to probabilities
    exp_preds = np.exp(preds)
    preds = exp_preds /np.sum(exp_preds)

    #randomly sample from preds. returns a one-hot vector [1,0,0] if 0 is the chosen index
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)    #returns the index of the chosen character

def generate_text(length, temperature):

    #pick a random place to start generation
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    #generated is currently random index: random_index + 40
    generated += sentence
    for i in range(length):
        #so we basically feed it one batch at a time, as in one 40-character thing at a time.
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            #x[0 (this is always 0 since we send it only sequence at a time), at the current step in the sequence, this character] = 1
            x[0, t, char_to_index[character]] = 1   #input tensor

        predictions = model.predict(x, verbose = 0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character

    return generated


print('----------0.2-----------')
print(generate_text(300, 0.2))
print('----------0.4-----------')
print(generate_text(300, 0.4))
print('----------0.6-----------')
print(generate_text(300, 0.6))
print('----------0.8-----------')
print(generate_text(300, 0.8))
print('----------1.0-----------')
print(generate_text(300, 1.0))