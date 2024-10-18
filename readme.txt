The dataset can be found in the links below
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip 
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

The Flickr8k_Dataset.zip contains the images and Flickr8k_text.zip contains the titles 
lemmatized or not as well as the train test and validation names.

Τα αρχεία που θα χρησιμοποιηθούν είναι:

1)Flickr_8k.devImages.txt contains the validation names.

2)Flickr_8k.testImages.txt contains the test names.

3)Flickr_8k.trainImages.txt contains the training names.

4)Flickr8k.lemma.txt contains 8000 image names as well as 5 lemmatized titles for each image (40000 in length)

The following libaries are used (will add a requirements.txt later): pandas , pickle, numpy, nltk, keras 
tqdm, torchvision και tensorflow  

Follow the next 3 steps for execution
1) Run CreateDataset.py which creates and saves all the datasets in the location
datasets\newdatasets 8 will be created in total.

2) Run Train_Language_Model.py which creates and trains the neural   
RNN model. Saves all the epochs in the usefulmodels file.

3) Run Compute_Scores.py which calculates the blue-4 score and prints the title and the image of the test dataset.
Can also run the function blueScore which computes the mean score of bleu-1-2-3-4 or Meteor Score on the train dataset

Finally you can test the model using your own images by saving them in the Images folder
