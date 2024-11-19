import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from nltk.translate import  bleu_score
from nltk.translate.meteor_score import meteor_score

#decorer 
modeldec=keras.models.load_model("usefulmodels/model.keras")

def main():
    #encoder

    path="datasets/newdataset/"
    #Load libraries
    with open(path+"words_to_indices.pkl", "rb") as pickle_f:
        words = pickle.load(pickle_f)
    with open(path+"indices_to_words.pkl", "rb") as pickle_f:
        indices = pickle.load(pickle_f)
    with open(path+"test_features.p", "rb") as pickle_f:
        test_features = pickle.load(pickle_f)
    with open(path+"test_captions.p", "rb") as pickle_f:
        test_captions = pickle.load(pickle_f)
    shape=len(test_features["2654514044_a70a6e2c21.jpg"])
    

    
    test(5,test_features,test_captions,words,indices,shape)
    #bleuScore(5,val_features,val_captions,words,indices,shape)

    # to try other examples found in Images folder( you can add your own ) use the code below
    #modelenc=ResNet50(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3))
    #pathToImages="Images/dogs2.jpg"
    # features=extract_features(pathToImages,modelenc)
    # print(beam_search(features,5,words,indices,shape))

def test(k,features,captions,words,indices,shape):
    counter=0
    path="flickr/Flicker8k_Dataset/"
    for img_id in tqdm(features):
        img=cv2.imread( path + img_id ) 
        counter+=1
        image=features[img_id]
        reference=[]
        for caps in captions[img_id]:
            list_caps=caps.split(" ")
            list_caps=list_caps[1:-1]
            reference.append(list_caps)
        candidate=beam_search(image,k,words,indices,shape)
        print("===========")
        print("PREDICTED CAPTION: ",candidate)
        print("===========")
        for j in reference:
            print(j)
        #bleu-4
        score = sentence_bleu(reference, candidate ,smoothing_function=bleu_score.SmoothingFunction().method7)
        print(score)  
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()   

def bleuScore(k,features,captions,words,indices,shape):
    counter=0
    tot_score=0
    n=1/4
    for img_id in tqdm(features):
        counter+=1
        image=features[img_id]
        reference=[]
        for caps in captions[img_id]:
            list_caps=caps.split(" ")
            list_caps=list_caps[1:-1]
            reference.append(list_caps)
        candidate=beam_search(image,k,words,indices,shape)
        #Bleu-4
        score = sentence_bleu(reference, candidate,weights=(n,n,n,n),smoothing_function=bleu_score.SmoothingFunction().method7)
        #calculate Meteor Score
        #score = meteor_score(reference, candidate)    
        tot_score+=score
    avg_score=tot_score/counter
    print("Avg_score: ",avg_score)
    return avg_score

def beam_search(photo,k,words,indices,shape):
    max_length=40
    photo=photo.reshape(1,shape)
    in_text='<start>'
    sequence = [words[s] for s in in_text.split(" ") if s in words]
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    y_pred = modeldec.predict([photo,sequence],verbose=0)
    predicted=[]
    y_pred=y_pred.reshape(-1)
    for i in range(y_pred.shape[0]):
        predicted.append((i,y_pred[i]))
    predicted=sorted(predicted,key=lambda x:x[1])[::-1]
    b_search=[]
    for i in range(k):
        word = indices[predicted[i][0]]
        b_search.append((in_text +' ' + word,predicted[i][1]))
    for idx in range(max_length):
        b_search_square=[]
        for text in b_search:
            if text[0].split(" ")[-1]=="<end>":
                break
            sequence = [words[s] for s in text[0].split(" ") if s in words]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            y_pred = modeldec.predict([photo,sequence],verbose=0)
            predicted=[]
            y_pred=y_pred.reshape(-1)
            for i in range(y_pred.shape[0]):
                predicted.append((i,y_pred[i]))
            predicted=sorted(predicted,key=lambda x:x[1])[::-1]
            for i in range(k):
                word = indices[predicted[i][0]]
                b_search_square.append((text[0] +' ' + word,predicted[i][1]*text[1]))
        if(len(b_search_square)>0):
            b_search=(sorted(b_search_square,key=lambda x:x[1])[::-1])[:5]
    final=b_search[0][0].split()
    final = final[1:-1]
    return final

def extract_features(path,modelenc):
    features = {}
    #load and resize the image to the resnet prerequisites
    img=image.load_img(path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #preprocess the given image according to ResNet's preprocessing method
    x = preprocess_input(x)
    features = modelenc.predict(x)
    return features

if __name__ == "__main__":
    main()