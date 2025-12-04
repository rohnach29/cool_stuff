import numpy as np
import keras
from keras import layers
import tensorflow as tf
with open("fortunes_10000.txt", "r") as f:
    text = f.read().lower()

li = sorted(list(set(text)))

char_to_index = dict((i,c) for (c,i) in enumerate(li))
index_to_char = dict((c,i) for (c,i) in enumerate(li))
print(char_to_index)

SEQ_LENGTH = 40
STRIDE = 3
sentences, next_letter = [], []
for i in range(0, len(text) - SEQ_LENGTH - 1, STRIDE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_letter.append(text[i + SEQ_LENGTH])


#what are x and y here?
X = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
Y = np.zeros(len(sentences), dtype=np.int32)

for i in range(len(sentences)):
    for j in range(SEQ_LENGTH):
        X[i][j] = char_to_index[sentences[i][j]]

    Y[i] = char_to_index[next_letter[i]]

print(sentences[0], X[0])
print(next_letter[0], Y[0])

# Now! onto transformers. We can't really use a Sequential() here, because:
#1) Keras doesn't have a Transformer() cell block, so need to build it manually
#2) when building it manually, we have many residual connections: x_in -> some layer -> x_out. x_out += x_in.

V = len(char_to_index)   # vocab size
T = SEQ_LENGTH           # 40
D = 128   # embedding width

#each input is a sequence of T ints. during runtime, this changes to (B,T)
#it's inherent to keras.Input to change this to (B,T) -- this doesn't happen with the constant tensor we define for pos_embedding, so we need to manually add a dimension there.
input = keras.Input(shape=(T,), dtype=np.int32) #the model expects input of (batch_size, SEQ_LENGTH. batch size is flexible. But this basically sets a template for what the input will be)
emb_layer = layers.Embedding(input_dim =V , output_dim = D) #this is our word embedding "lookup table": we have V characters, and each chatacter is defined by a 128-dimension positional vector
embedded_input = emb_layer(input)   #apply the Embedding layer on the input. End with (B,T,D).
embedded_input = embedded_input * tf.math.sqrt(tf.cast(D, tf.float32))  # need to normalize, because we add word embeddings to positional embeddings, so don't want one to dominate.
#nevermind, apparently we WANT the word embedding to dominate -- we only multiple word embedding by root(D) and not position, because we want word embedding to be "louder"
#why scale at all?:
# 1) want word embeddings to be louder, because they are more relevant than positional embeddings
# 2) if the values are too small, dot product can become ~0, and softmax will be uniform. want bigger values so that attention gives us some meaningful output.


#now onto the positional embedding part
#here, it's good practice to have a max_len in the positiona_embeddings lookup table, so that if we increase SEQ_LENGTH later, don't have to change the whole model
max_len = 1024
#we don't need an Input() part here. why?:
# 1) positional encoding doesn't really rely on the input -- it just relies on the SEQ_LENGTH, so we don't really need to care about the input here
# 2) We add it to the vector embedding right after, so it basically includes the input anyways.
pos_indices = tf.range(T) #makes a tensor of (T,)
pos_embedding = layers.Embedding(input_dim = max_len, output_dim = D)   #the shape of this is (max_len, D)
pos_emb = pos_embedding(pos_indices)  #the shape of this is (T,D). this just takes the first T rows of the embedding
pos_emb = tf.expand_dims(pos_emb, 0) #finally, this is (1,T,D) -> word embedding layer is (B, T, D), and need to add them.

x = embedded_input + pos_emb
#layers.Dropout(0.1) instantiates a Dropout object, and then we call x
#why is the format so weird? Isn't it Object.function(x)?
#Python has __call__, which makes the obejct itself callable. So here, the Dropout object itself is callable. Weird.
x = layers.Dropout(0.1)(x)

#for next time:
# understand the transformer block

def transformer_block(x, d_model, num_heads = 4, d_ff = None, dropout = 0.1):
    if d_ff == None:
        d_ff = 4 * d_model

    #so apparently during word embedding and position embedding, some tokens can end up with a mean across all dimensions.
    #thus, we need to normalize each d-dimensional vector so that they each have mean=0 and variance=1
    #why do we need epsilon=1e-6? in the rare case that standard deviation is 0, it ensures we don't divide by 0.
    y = layers.LayerNormalization(epsilon=1e-6)(x)

    # i am kinda confused about why Q,K,V are 32 dims each, but here is what I think it means
    # it doesn't break the input into 32 dims, but it's just that each head combines the learnings of all 128 dimensions into 32 dimensions
    # eg dimension 1 could mean "looks like end of word", dimension 7 could mean "is vowel + near space"
    # and then to get a vector of original (B,T,D) size, we put it all together.
    #4 heads in parallel, each compressing the 128 dims differently, let the model:
    #---for eg---
    # Head 1: specialize in local patterns
    # Head 2: specialize in long-range relations
    # Head 3: focus on punctuation/spacing
    # Head 4: focus on stylistic cues
    y = layers.MultiHeadAttention(
        num_heads = num_heads,
        key_dim = d_model // num_heads,
        dropout = dropout   #this refers to attention weights during softmax
    )(y, y)

    y = layers.Dropout(dropout)(y)
    x = layers.Add()([x,y])

    #what is this?
    #basically, after attention, need to retrain each point based on what it has learned
    #intuition: attention is a discussion with others, FFN is a discussion with the self.
    #learns how to use the attention values properly
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    #scale it up to 4 * num_dim to learn a richer, more complex version of things. 
    #another advantage of FFN: transformers are quite linear, so we introduce non-linearity here with an activation function
    #gelu is gaussian. found that it's better for transformers than relu
    y = layers.Dense(d_ff, activation="gelu")(y)
    #then, we squash it back to 128 dims, because we need the residual skip connection (add the values to previous ones)
    #why not use an activation function here?
    #it's like adding apples to squashed apples. activation would distort the signal.
    #we need to add something similar to x for residual skip connection, not an entirely different beast
    #distort signal? with tanh, squashes to -1 to 1, so would lose magnitude. relu 0 to 1, would lose negative numbers and bias it towards positives.
    y = layers.Dense(d_model)(y)
    y = layers.Dropout(dropout)(y)
    x = layers.Add()([x,y])

    #transformers rely heavily on residual skip connections.
    #why? if stacking layers, if the layer does nothing useful, the signal going forward could get mangled.
    #instead of reinventing x everytime, we just keep what information we have and add it to x
    #basically x + a delta, so that we still preserve useful information even if the layer doesn't do anything useful.
    #instead of "reinvent yourself completely every layer", it's "carry yourself forward, plus add a little adjustment."

    return x

#stacking 3-4 transformer blocks
#why even stack? deeper models -> better learning. just one block could be shallow.
for i in range(3):
    x = transformer_block(x, D, 4, 4 * D, 0.1)

x = layers.LayerNormalization(epsilon=1e-6)(x)

#this "last" thing is a bit confusing, but here's what I understood:
#in GPTs, every step needs to predict the next token. Their final tensor is (B,T,V)
#eg. input:   "the cat sat on the ma"   targets: "he cat sat on the mat"
#thus, they use a "causal mask" so that at each token, attention can only be focused behind. no "cheating" by looking ahead, basically
#in our case, we are given 40 chars and need to predict the 41st. In other words, our answer isn't given to us in the input itself -> we need to predict the answer
#in GPTs, they need to predict the next token, but the next token is already given to them. so, they can only look behind to prevent them from cheating
#long story short, I think it is more convention and it makes intuitive sense too -- I do want to test it out though:
#Since we're not using a causal mask and each character has context of the rest, what if we point "last" to a random character in the middle?

#hmm here's what it is -> if we take a middle token, say 3, it has already peeked at the tokens ahead and that is "cheating" in a sense
#it already knows the answers ahead
#in my head, it was "token 3 = token 40", but that's not true -- each thing has positional encoding.
#so token 3 asks: “given that I’m at position 3, who’s important to me?”
#token 40 asks: “given that I’m at position 40, who’s important to me?”
# the whole network is trained so that the last token’s query is the one that learns to summarize the context for predicting the next character.

#but what if we get the model/network to be trained so that the 3rd token's query is the one that learns to summarize the context? wouldn't last = [:,2,:] do this?
#yes! you could technically train on the last character!!! but it's just a bit awkward
#in BERT -> [CLS] token, which is basically a "summarizer" of everything
last = x[:,-1,:]
logits = layers.Dense(V, activation="softmax")(last) #gives us (B,V), which is exactly what we want!!!!

#oh wow so this wires together the whole flow and connects everything
#this connects input (where it starts) to logits(where it ends), and thus everything in between! it thus makes the whole model by connecting input to output
model = keras.Model(inputs=input, outputs=logits)

# compile with Adam + cross entropy loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# early stopping to avoid overfitting
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X, Y,
    validation_split=0.1,
    batch_size=128,
    epochs=30,
    callbacks=[early_stop]
)

# ---- save model ----
model.save("transformer_fortune.keras")

#to run this next part, comment out the above code

model = tf.keras.models.load_model("transformer_fortune.keras")

def sample(preds, temperature=1.0):
    #logits is of size (B,V)
    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature
    exp = np.exp(preds)
    preds = exp / np.sum(exp) #flattens into 1D and adds everything

    prob = np.random.multinomial(1, preds, 1)   #prob is a one-hot vector with the chosen sample -> [0,0,1,0....]
    return np.argmax(prob)


def generate(length, temperature):
    start = np.random.randint(0, len(text) - SEQ_LENGTH)
    generated = text[start: start + SEQ_LENGTH]

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



