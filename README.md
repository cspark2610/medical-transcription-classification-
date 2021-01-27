# Classification of 13 Medical Specialties Through Transcriptions Notes Using ScispaCy, CountVectorization, and Tf-Idf 

## Dataset
Dataset retrieved from kaggle.com - originally from mtsamples.com
Contains 500 rows of de-identified medical information, which includes description, medical specialty, sample name, transcription, and keywords. For the scope of this project - only transcription and medical specialy will be used.


## Approach

Extract medical_specialty and transcription columns.
Extract classes that will be used from our dependent var, medical_specialty.

Split dataset into training and testing sets.

Extract features using Count Vectorization(CV) and Tf-Idf(tf).

Apply Truncated SVD onto term count/tf-idf matrices (LSA).

EDA of  outputs - examine texts, possible patterns, visualize t-SNE plots, etc.

Get baseline models for the following seven classifiers:

Stochastic Gradient Descent Classifier
Multinomial Logistic Regression
Logistic Regression OVR
AdaBoosted Decision Trees Classifier
Linear Support Vector Classifier OVR
K-Nearest Neighbors Classifier
LightGBM Classifier 

* 'Medical transcriptions' or 'documents' will be used interchangably.

### Pre-Vectorization: Filtering Classes (Dependent Variable Groups)
There are no missing values for medical_specialty and 0.66% missing for transcription, which were dropped.

PLOT 

The figure shows evidence of very high class imbalance which will be an issue for classification. It also shows that 'Surgery' has far more transcripts than any other classes and a good amount of classes with very low transcripts.

After looking into 'Surgery', it appears that this class is a combination of general surgeries, and specialized surgeries and procedures of other medical classes. For reasons of potential overlapping with other categogies as well as the enormous difference in transcripts it has compared to the other 39 classes, 'surgery' will be dropped. 

In addition, the aim of this project is to classify medical specialties; therefore 'ambulatory' classes such as 'OfficeNotes', 'Letters', 'IME-QME-WorkCompetc., and, 'Consult-HistoryandPhy' will be dropped.

Lastly, a threshold needs to be determined for having a certain number of transcripts. After threshold attempts at 25 and 50, I found 75 to be a good estimate.

In total, 13 unique medical specialties were selected for text preprocessing.

FIGURE

* Note: I've attempted to salvage some of the lower classes by also using resampling techniques  SMOTE, ADASYN, SMOTEEN, and SMOTETOMEK, however, these methods were ineffective and only led to the addition of more noise. 

### Text Preprocessing

Two text preprocessing functions were used for comparison. Both functions contained the same preprocessing steps except text_scispaCy_preprocess includes scispaCy's package, 'en_core_sci_sm', which is used to help extract biomedical texts and embeddings, in addition to other functionalities.

Prepocessing steps:

Transcripts converted into list of words, lowercased, and fixed contractions (i.e., that's -> that is). Punctuations, stopwords, digits, special characters, and outliers  ('mmddyyyy','abc', etc.) were removed. Texts were lemmatized and texts with less than 2 characters were removed.

### Feature Extraction - Count Vectorization (CV) and term frequency inverse document frequency (Tf-IDF)(TF)

Dataset were split into training and testing sets using 'train_test_split' which were stratified by classes at 20% test size.

Count Vectorization and Tf-Idf were set to using upto 3-grams, meaning unigrams, bigrams and trigrams would be extracted. Max features were set to 5000 unique texts; higher number of features used would lead to better classification rates, however, by doing so, will require higher computing power and memory.

Texts with atleast 4 and 5 document appearances, tf and cv respectively, and none appearing in 95% of total documents were extracted. 

For both methods, only training sets were fitted, followed by transformation of training and testing sets. Fitting and transforming an all inclusive set would enable data leakage.

### Truncated SVD 

Truncated SVD was applied to text count/tf-idf sparse matrices - a conventional process known as latent semanatic analysis (LSA). The benefit of LSA is that it transforms the document-term/sparse matrices into a "semantic" space of lower dimensionality and uses singular value decomposition to find hidden or latent meanings through texts and docs; therefore, the proportional variance explained by the sparse matrices can be explained with less features(texts).

The number of features were reduced from 5000 to 812 for CV training set, and 5000 to 1181 for TF training set. 

        Initial Feature Training Set -->{(2380, 5000)}
        Initial Feature Testing Set -->{(596, 5000)}
        **************************************************
        Modified Feature Training Set -->(2380, 812)
        Modified Feature Testing Set -->(596, 812)
        **************************************************
        Explained proportional variance: 94%
        Estimating number of components...
        1181 components needed to explain 95.0% variance.
        
The LSA outputs were normalized. Since, the outputs of TfIdf and Countvect are normalized, LSA/SVD results are not. 

### t-SNE Plots and Text EDA


*
From the barplot it is evident that mainly surgical/proecdural based classess, Orthopedics/Neurosurgery have higher text averages per transcription; this is most likely due to the preciceness and detailed nature required for documenting surgeries. And, Radiologist are at the bottom sinc e their field tends to vear towards more succint explanations/diagnoses of imaging scans and MRIs. 

* Ironically, having shadowing experiences within these three fields, I believe I can, very slightly, attest to this speculation.

Chi-squared tests were conducted to show correlated texts per class, and examine anuy discrepancies between countvectorized and tfidf values.

t-SNE scatter plots of Tf-idf and CV training data were produced with metric 'cosine' .
The visualizations provided very interesting insight into the allocation of medical class values on a 2-D plot. 

Both tf-idf and cv plots showed positive signs of slightly uniform clusters for certain classes, however, there is also evidence a large area composed of overlapping values from different classes - mainly Cardiovascular/Pulmonary and General Medicine.

Classifiers will most likely be able to fit the uniformed clusters fairly easily, but will face difficulties in distinguishing the central area comprised of various classes. 


Word Clouds! Because everyone loves them.
Word clouds show that most common terms are what most of us think when it comes to medicine/healthcare such as "patient, left, right pain, procedur, etc...".



## Baseline Modeling

All classifiers were set at default settings for baseline results.

After visuzliing t-SNE plot and estimating similiarties between several medical classes.
I decided to use boosting and one vs rest (OVR) approach might have some promise in distinguishing the the main overlapping areas; KNN will not be able to improve much more than its values at baseline due to to the nature of its alghoritm.

The scispaCy model did show higher metrics compared to pre-scispaCy metrics. And, considering the other functionalities the biomed sciscpaCy package contians, I assumme it is possible to improve metrics even further; for example, I did not convert medical abbreviations into words (MI = myocardialinfarction, BMI=bodymassindex, etc).

Tfif mean F1-Scores are higher than CV scores.
The inclusion of spaCy package seems to have very little effect on CV compared to tf.

*TfIdf Classification Reports 
Stochastic Gradient Descent:
Strength: Overall 
Weakness: radiology and neurology
One vs Rest Logistic Regression: 
strong in classifying classes with larger counts, 
weak in radiology
Multinomial Logistic Regression: 
same as log reg OVR
Linear Support Vector Classification OVR: 
very poor radiology nad neurology and neurosurgery,
others strong
Ada BoostedTrees: 
weak in generally all classes besides opthalmology and orthopedics but it is expected since max_depth =1
K-Nearest Neighbors Classifier:
strong in every category exceot radiology and neurosurgery
Light GBM: 
very weak in radiology, neurology, hematology, nephrology, neurosurgery, otherwise robust

*CV Classification Reports 
Stochastic Gradient Descent: 
very weak distinguishing radiology and neurology and neurosurgery
One vs Rest Logistic Regression: 
strong in classifying classes with larger counts, weak in radiology
Multinomial Logistic Regression: 
same as log reg OVR
Linear Support Vector Classification OVR:
very poor radiology and weak in neurology and neurosurgery, others strong
Ada BoostedTrees: weak in general, but it is expected since max_depth =1
K-Nearest Neighbors Classifier:strong in every category exceot radiology and neurosurgery
Light GBM: very weak in radiology, neurology, hematology, nephrology, neurosurgery, otherwise robust

Main issue: radiology and neurology/neurosurgery (going back to t-SNE plot, this is expected; generalmedicine and cardiovascular/pulmonary were not as problematic as I presumed they would be. 

## Hyperparameter Tuning using GridSearchCV and RandomSearchCV with 5-fold stratified cross-validaiton

RandomSearchCV is much faster than brute force GridSearchCV for tuning hyperparameters. Therefore using RandomSearchCV to narrow down optimal hyperparameters, I utilized GridSearchCV to confirm.

For LightGBM, I used Optuna, a hyperparameter optimization framework, for finding optimal hyperparameters at learning rate of 0.01. It calculates number of estimators, feature_fraction, num_leaves,  To avoid overfitting, the final step employs several regularization methods to best adjust the multi_logloss. 

## Results
Stochastic Gradient Descent Classifier with Tf-Idf vectorization produced highest F1-score, %. 

Logistic Regression OVR 

Multinomial Logistic Regression came close at second best score, %.



Tf-Idf produced higher F1-scores 

Linear Support Vector Classifier: Countvectorized tuned model outperformed Tf-Idf models.

LightGBM Classifier: Tfidf tuned model produced highest
AdaBoostedTree Classifier: Benefited greatly from hyperparameter tuning and produced a F1-score of 

## Conclusions
Including scispaCy's biomed package helped improve metrics for all classifiers with the exception of KKN Clf. It would be interesting to look deeper into what other funcionalities it possess that can possibly improve classification rates even more. Initially, I hypothesized GeneralMedicine or Cardiovascular/pulmonary class to be the biggest obstacle for this project; since the nature of general medicine being more generalized would encompass factors that would overlap with other specialties and Cardiovascular/pulmonary, due to having the highest transcript count, as well as it being a medical field that has numerous associations with morbidities in other specilties. KNN clf, as I thought, was relatively consistent even without tuning, it was on par with other tuned models, since specialties would assumed to cluster as long as feature extraction worked properly; but there is an extent to KNN clf's capability to distinguish overlapping classes. As all classifiers did with radiology and neurology. Even SGD clf, the highest performing classifer, was only able to achieve an F1-score of 33%, but, Neurology did increase signficantly to 59%. 

Neurology and Radiology both comprise of similiar procedures such as MRIs and CT scans, in addition to neuroradiology, a specialty that wasn't accounted for with this dataset.

In conclusion, with these findings, I believe by implementing more text preprocessing methods such as conversion of medical jargon, acronyms, partitioning transcripts by common headers such as, "SUBJECTIVE, HISTORY, CHIEF COMPLAINT", and classifying based on these subsets; moreover, inclusion of more classes or breakdown of more class to have classes like "neuroradiology, interventional radiology, etc". And, of course, always, more data. 


## limitations
However, I have not attempted downsampling  majority classess, simply due to having already dropped a great amount of data, but it may be possible to partition and reallocate 'surgery' data by identifying specialty-based procedures. 

dropping
