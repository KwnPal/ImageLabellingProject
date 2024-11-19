from numpy import array
import pickle
from tqdm import tqdm
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import add
from keras.layers import Embedding


max_length=40
def main():
    pathe="datasets/newdataset/"
    shape=[]
    
    with open(pathe+"train_captions.p", "rb") as pickle_f:
        train_captions = pickle.load(pickle_f)
    with open(pathe+"words_to_indices.pkl", "rb") as pickle_f:
        words_to_indices = pickle.load(pickle_f)
    with open(pathe+"train_features.p", "rb") as pickle_f:
        train_features = pickle.load(pickle_f)

    #forming dictionaries containg mapping of words to indices and indices to words
    shape=train_features["2513260012_03d33305cf.jpg"].shape
    print(shape)
    #forming dictionary having encoded captions
    train_encoded_captions={}
    for img_id in train_captions:
        train_encoded_captions[img_id]=[]
        for i in range(5):
            train_encoded_captions[img_id].append([words_to_indices[s] for s in train_captions[img_id][i].split(" ")])
    
    for img_id in train_encoded_captions:
        train_encoded_captions[img_id]=pad_sequences(train_encoded_captions[img_id], maxlen=max_length, padding='post')
    
    vocab_size=len(words_to_indices)
    
    #Initialize the model architecture
    input_1=Input(shape=(shape))
    dropout_1=Dropout(0.2)(input_1)
    dense_1=Dense(256,activation='relu')(dropout_1)

    input_2=Input(shape=(max_length,))
    embedding_1=Embedding(vocab_size,256)(input_2)
    dropout_2=Dropout(0.2)(embedding_1)
    lstm_1=LSTM(256)(dropout_2)

    add_1=add([dense_1,lstm_1])
    dense_2=Dense(256,activation='relu')(add_1)
    dense_3=Dense(vocab_size,activation='softmax')(dense_2)

    model=Model(inputs=[input_1,input_2],outputs=dense_3)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    epochs=50
    no_of_photos=5
    print(len(train_encoded_captions))
    steps=len(train_encoded_captions)//no_of_photos
    for i in range(epochs):
        generator=data_generator(train_encoded_captions,train_features,no_of_photos,vocab_size)
        model.fit(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        model.save("usefulmodels/model"+str(i)+".keras")
    



def data_generator(train_encoded_captions,train_features,num_of_photos,vocab_size):
    X1, X2, Y = list(), list(), list()
    n=0
    for img_id in tqdm(train_encoded_captions):
        n+=1
        for i in range(5):
            for j in range(1,max_length):
                curr_sequence=train_encoded_captions[img_id][i][0:j].tolist()
                next_word=train_encoded_captions[img_id][i][j]
                curr_sequence=pad_sequences([curr_sequence], maxlen=max_length, padding='post')[0]
                one_hot_next_word=to_categorical([next_word],vocab_size)[0]
                X1.append(train_features[img_id])
                X2.append(curr_sequence)
                Y.append(one_hot_next_word)
        if(n==num_of_photos):
            yield [[array(X1), array(X2)], array(Y)]
            X1, X2, Y = list(), list(), list()
            n=0


if __name__ == "__main__":
    main()