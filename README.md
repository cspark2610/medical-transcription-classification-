# medical-transcription-classification-

##
Dataset from mtsamples.com
Contains 500 medical transcriptions from 40 different medical specialties.

Objective is to assess classification rates of medical specialties by medical transcriptions.

Transcriptions will undergo 2 feature extraction methods, Count Vectorization(CV) and Tf-Idf(tf).

The following classifers will be used to:

Stochastic Gradient Descent
Multinomial Logistic Regression
One vs Rest Logistic Regression
Ada BoostedTrees
Support Vector Machine
K-Nearest Neighbors Classifier
Light GBM 

* 'Medical transcriptions' or 'documents' will be used interchangably.

### Pre-Vectorization: Filtering Classes (Dependent Variable Groups)
There were no missing values for 'medical_specialty' and, only, 0.66% missing for transcription, which were dropped (rows).

The 'number of transcriptions per medical specialty' plot shows that 'Surgery' has the greatest amount of documents, by far at 1,088, and shows evidence of very high class imbalance which will pose as an obstacle for classification. 

Therefore, after, digging further into 'Surgery', it appears that this class is more of a generalized category for both surgery related transcriptions and procedures that overlap with our other specialty groups, that are not 'Plastic' or 'Neurosurgery' (these exist as seperate classes). Therefore, despite the high loss of data, for several reasons the 'Surgery' group was filtered out.

Additionally, the aim of this project is to classify medical specialties as in specialized fields of medicine, thus class such as 'OfficeNotes', 'Letters', 'IME-QME-WorkCompetc., and, arguably, 'Consult-HistoryandPhy' (our second highest data class).

Lastly, the number of classes that present low number document counts were dropped. After, several poor attempts, I decided that classes should have atleast 75 documents. 

* I've attempted to salvage some of the lower classes by using SMOTE, ADASYN, SMOTEEN, and SMOTETOMEK, however these led to mainly resampling bias or noise, thus I made the decision to filter classes under 75 docs. However, I have not attempted downsampling  majority classess, simply due to having already lost a great amount of data from dropping 'surgery', but it may be possible to breakdown and redistribute 'surgery' data by identifying and extracting type of procedures and resample these data back but into their appropriate specialty class, however, these are options for future projects.

In total, 13 unique medical specialties were selected for text preprocessing.
### Text Preprocessing
Documents or transcriptions were passed through my preprcoessing function composed of several transformations.

Initially, documents were split and converted into a list of words or 'tokens' that were 'lowercased' and converted contractions into tokens (i.e., that's -> that is).
Followed by punctuation, digits, special characters, and certain outliers  ('mmddyyyy','abc', etc.) removal.

Texts were then lemmatized, removed if were 'stopwords', stemmed and filtered if character length were less than 2.

* Later on, I came across scispaCy's package, 'en_core_sci_sm', that contained biomedical text embeddings, abbreviation linkages, and other specialized functionalities that seemed to be very useful within biomedical nlp field, but beyond the scope of this project; but, I included the package, for its' main capability to identify biomed terms, into my text preprocessing function to evaluate discrepenacies in output.

### Feature Extraction - Count Vectorization (CV) and term frequency inverse document frequency (Tf-IDF)(TF)

For feature extraction, I developed a function that will begin with 'traintestsplit', stratified by classes at 20% test size. So, that CV and TF will only be fit to the training set, and fitted to all inclusive set, which would lead to data leakage. Transformations for training and testing sets were performed by vectorizers fitted only on training sets. 

Both CV and TF were conducted using 3-grams aka trigrams and set to analyze by 'word'.
The number of max features was set to 5000 unique texts; although, increasing the number of features would be conducive for better classification rates, however, it also requires higher computing power and memory.

Texts with atleast 4(TF) or 5(CV) document appearances, and none appearing in 95% of total documents were extracted. 

My theory is that tf-idf will out perform count vectorized texts, causeee

### Truncated SVD 
Optional step:

Applying truncated SVD onto sparse matrices, the outputs of tfidf and cv, is a conventional process known as latent semanatic analysis (LSA), simply because it "transforms the document-term/sparase matrices into a "semantic" space of lower dimensionality (ref @ url below), or in other words can explain the equivalent or similiar proportional variance that can be explained by the DTM matrices without TSVD, but with a reduced number of features(texts), due to being able to interpret values semantically.

* https://vitalv.github.io/projects/doc-clustering-topic-modeling.html <- reference; explains LSA as well as other info very well, highly recommend reading 

With TSVD, the number of features were be able to reduce from 5000 to 812 for my CV training set, and 5000 to 1181 for TF training set. Reducing the number of features allowed for more efficient work to be done, by not having to wait on your comuter/notebook run.

        Initial Feature Training Set -->{(2380, 5000)}
        Initial Feature Testing Set -->{(596, 5000)}
        **************************************************
        Modified Feature Training Set -->(2380, 812)
        Modified Feature Testing Set -->(596, 812)
        **************************************************
        Explained proportional variance: 94%
        Estimating number of components...
        1181 components needed to explain 95.0% variance.
        
 ### t-SNE Plots and Text Explorations
t-SNE Plots of Tf-idf and CV training set data presented good and bad clusters. My assumption is that medical specialties classess possess values that are both specialized and portions of general medicine. The clusters that are far easier to distinguish from others characterize the specialized portion, while, the other portion becomes integrated with other generalized portions from other specialties, and, evidently, general medicine. So, prior to any classificaiton, I believe it will produce decent base metrics, however, hyperparameter tuning and other adjustments will be difficult to untangle this "generalized med portion". KNN Classifier, specifically, should output well for base model but I doubt there will be much increase by tuning.

The average text per transcription by medical specialty bar plot shows, surgical procedures Neurosurgery and orthopedic (often procedural based) are more explicit thus contain higher average of text per transcript. Having shadowed my uncle who is a Radiologist, I am not surprised that they rank lowest. Often, their dictations are succint and straight to the point.

Word Clouds! Because everyone loves them.
Unfortunately, although the word clouds do represent most common words, by frequency, it does not account for weights like idf. So, evidently, we get terms like "patient, left, right pain, procedur, etc..." that we often associate when it comes to medicine. 

For further analysis, I performed chi-squared correlation on tfidf training set to output most correlated texts to their specialty by 1,2,3 grams. The correlated terms were very fitting and explained the unqiueness of its corresponding specialty.

* I thought of using the chi-squred outputs to be used for resampling minority classes-assist in class imbalance, but I am not sure how much bias I would be introducing into the overall dataset, even though conventional resampling methods use existing terms, unlike SMOTE. This is something I will look more into.

## Baseline Modeling
Finally, modeling!

To obtain baseline results, I ran all classifiers at default settings.
The reasons for the classifiers, I've chosen were mainly, after visuzliing t-SNE plot.
My reasoning was that perhaps boosting and one vs rest (OVR) approach might hold the most promise in distinguishing the 'generalized blob'; and, that KNN will not be able to improve much more than its values at baseline due to to the nature of its alghoritm.

The scispaCy model did show metric improvements compared to pre-scispaCy metrics. And, considering the other functionalities the biomed sciscpaCy package contians, I assumme it is possible to improve metrics even further; for example, I did not convert medical abbreviations into words (MI = myocardialinfarction, BMI=bodymassindex, etc).

Tfif mean F1-Scores are higher than CV scores; and the inclusion of spaCy package seems to have very little effect to CV.


