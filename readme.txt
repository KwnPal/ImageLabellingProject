Το dataset που χρησιμοποιήθηκε μπορεί να βρεθεί στα παρακάτω links.
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip 
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

Το Flickr8k_Dataset.zip περιέχει τις εικόνες και το Flickr8k_text.zip περιέχει τους τίτλους 
(λημματοποιημένους και μη) και τα ονόματα των train test validation.
Τα αρχεία που θα χρησιμοποιηθούν είναι:

1)Flickr_8k.devImages.txt εμπεριέχει τα όνοματα για το validation.

2)Flickr_8k.testImages.txt εμπεριέχει τα ονόματα για το test.

3)Flickr_8k.trainImages.txt εμπεριέχει τα ονόματα για το train.

4)Flickr8k.lemma.txt εμπεριέχει τα 8000 ονόματα ονόματα και τους 5 λημματοποιημένους τίτλους της κάθες εικόνας
(σύνολο 40000)

Για να εκτελέσετε τον κώδικα θα χρειαστούν οι βιβλιοθήκες: pandas , pickle, numpy, nltk, keras 
tqdm, torchvision και tensorflow  

1)Εκτελείται το αρχείο CreateDataset.py που δημιουργεί και αποθηκεύει όλα τα λεξικά στο αρχείο
datasets\newdatasets θα δημιουργηθούν 8 στο σύνολο.

2)Εκτελείται το αρχείο Train_Language_Model.py το οποίο δημιουργεί και εκπαιδεύει το Νευρωνικό 
RNN μοντέλο. Αποθηκεύει όλες τις εποχές στον φάκελο usefulmodels.

3)Εκτελείται το αρχείο Compute_Scores.py που τρέχει την συνάρτηση test η οποία επιστρέφει 
blue-4 score, τίτλο και εικόνα του test dataset.

Ενναλακτικά τρέχει τη συνάρτηση blueScore που βρίσκει το μέσο όρο bleu-1-2-3-4
(μεταβάλλοντας τα βάρυ βρίσκουμε και το αντιστοιχο score πχ. weights=(n,n,n,0) n=1/3 μας δίνει το bleu-3) 
ή το Meteor Score του train dataset.

Επιπλέον μπορεί ο χρήστης να βάλει την δική του εικόνα στον φάκελο Images και να αλλάξει το 
pathToImages στο αντίστοιχο όνομα.
