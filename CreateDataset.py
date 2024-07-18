import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
import numpy as np
from nltk.corpus import stopwords
import string
from keras.preprocessing import image
from tqdm import tqdm
import nltk
import torchvision.transforms as transforms
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications.vgg16 import preprocess_input
from nltk.translate.meteor_score import meteor_score
def main():
        #Be careful with the name, may overwrite old datasets
        pathe="datasets/newdataset/"
        nltk.download("wordnet")
        
        patho="flickr/"
        image_tokens=pd.read_csv(patho+"Flickr8k.Lemma.token.txt",sep='\t',names=["img_id","img_caption"])
        train_image_names=pd.read_csv(patho+"Flickr_8k.trainImages.txt",names=["img_id"])
        test_image_names=pd.read_csv(patho+"Flickr_8k.testImages.txt",names=["img_id"])
        val_image_names=pd.read_csv(patho+"Flickr_8k.devImages.txt",names=["img_id"])
        image_tokens=preprocessCaptions(image_tokens)

        #creating train dictionary having key as the image id and value as a list of its captions
        train_captions={}
        for i in tqdm(range(len(train_image_names))):
                l=[caption for caption in(image_tokens[image_tokens["img_id"] == train_image_names["img_id"].iloc[i]].img_caption)]
                train_captions[train_image_names["img_id"].iloc[i]]=l
        for i in train_captions:
                print(train_captions[i])
        with open(pathe+"train_captions.p", "wb" ) as pickle_f:
                pickle.dump(train_captions, pickle_f )
        #creating test dictionary having key as the image id and value as a list of its captions
        test_captions={}
        for i in tqdm(range(len(test_image_names))):
                l=[caption for caption in(image_tokens[image_tokens["img_id"] == test_image_names["img_id"].iloc[i]].img_caption)]
                test_captions[test_image_names["img_id"].iloc[i]]=l

        with open(pathe+"test_captions.p", "wb" ) as pickle_f:
                pickle.dump(test_captions, pickle_f )
        #creating validation dictionary having key as the image id and value as a list of its captions
        validation_captions={}
        for i in tqdm(range(len(val_image_names))):
                l=[caption for caption in(image_tokens[image_tokens["img_id"] == val_image_names["img_id"].iloc[i]].img_caption)]
                validation_captions[val_image_names["img_id"].iloc[i]]=l
        
        with open(pathe+"validation_captions.p", "wb" ) as pickle_f:
                pickle.dump(validation_captions, pickle_f )

        model=ResNet50(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3))
        #getting image encodings(features) from resnet50 and forming dict train_features
        path=patho+"Flicker8k_Dataset/"

        train_features={}
        for image_name in tqdm(train_captions):
                img_path=path+image_name
                img=preprocess(img_path)
                features = model.predict(img)
                train_features[image_name]=features.squeeze()

        with open(pathe+"train_features.p", "wb" ) as pickle_f:
                pickle.dump(train_features, pickle_f )

        test_features={}
        for image_name in tqdm(test_captions):
                img_path=path+image_name
                img=preprocess(img_path)
                features = model.predict(img)
                test_features[image_name]=features.squeeze()

        with open(pathe+ "test_features.p", "wb" ) as pickle_f:
                pickle.dump(test_features, pickle_f )

        validation_features={}
        for image_name in tqdm(validation_captions):
                img_path=path+image_name
                img=preprocess(img_path)
                features = model.predict(img)
                validation_features[image_name]=features.squeeze()

        with open(pathe+ "validation_features.p", "wb" ) as pickle_f:
                pickle.dump(validation_features, pickle_f )

        # Setting hyper parameters for vocabulary size and maximum length
        all_captions=[]
        for img_id in train_captions:
                for captions in train_captions[img_id]:
                        all_captions.append(captions)
        
        all_words=" ".join(all_captions)
        print(len(all_words))
        unique_words=list(set(all_words.strip().split(" ")))
        print(len(unique_words))

        words_to_indices={val:index+1 for index, val in enumerate(unique_words)}
        indices_to_words = { index+1:val for index, val in enumerate(unique_words)}
        words_to_indices["Unk"]=0
        indices_to_words[0]="Unk"

        with open(pathe + "words_to_indices.pkl", "wb") as pickle_f:
                pickle.dump(words_to_indices, pickle_f )
        with open(pathe + "indices_to_words.pkl", "wb") as pickle_f:
                pickle.dump(indices_to_words, pickle_f )

def preprocess(img_path):
    img=image.load_img(img_path, target_size=(224, 224))
    img=image.img_to_array(img)
    img= np.expand_dims(img, axis=0)
    img=preprocess_input(img)
    return img 


def preprocessCaptions(image_tokens):
        #removing the #0,#1,#2,#3,#5 from the image ids
        #removing Caps,punctuation and stopwords
        stopwords=["s"]
        image_tokens["img_id"]=image_tokens["img_id"].map(lambda x: x[:len(x)-2])
        for i in tqdm(range(len(image_tokens["img_caption"]))):
                image_tokens["img_caption"][i] = image_tokens["img_caption"][i].lower()
                image_tokens["img_caption"][i]= "".join([j for j in image_tokens["img_caption"][i] if j not in string.punctuation])
                image_tokens["img_caption"][i]=image_tokens["img_caption"][i].split(" ")
                image_tokens["img_caption"][i]=[j for j in image_tokens["img_caption"][i] if j not in stopwords]
                image_tokens["img_caption"][i]=" ".join(image_tokens["img_caption"][i])
        image_tokens["img_caption"]=image_tokens["img_caption"].map(lambda x: "<start> " + x.strip() + " <end>")

        return image_tokens
if __name__ == "__main__":
    main()