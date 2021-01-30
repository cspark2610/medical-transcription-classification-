# Text Classifcation using CountVectorization, TF-IDF, scispaCy

* Dataset: de-identified public source medical data from mtsamples.com
* Dataset details: contains 500 rows of de-identified medical information and 40 unique medical specialty classes, and also includes descriptions, sample names, and keywords; however we will only use "medical_specialty" and "transcription" columns for this project

## Approach
plann


## Objective: Classify medical transcript notes into predefined medical specialties 


### Classification Algorithms:
* Stochastic Gradient Descent Classifier OVR
* Logistic Regression Multinomial and OVR
* AdaBoosted Decision Trees Classifier
* Linear Support Vector Classifier OVR
* K-Nearest Neighbors Classifier
* LightGBM Classifier 



Starting off, I extracted  and removed missing values of only 0.66%.
The 40 unique classes contained within the "medical_specialty" variable are:

                ['Allergy/Immunology' 'Bariatrics' 'Cardiovascular/Pulmonary' 'Neurology'
                 'Dentistry' 'Urology' 'GeneralMedicine' 'Surgery' 'Speech-Language'
                 'SOAP/Chart/ProgressNotes' 'SleepMedicine' 'Rheumatology' 'Radiology'
                 'Psychiatry/Psychology' 'Podiatry' 'PhysicalMedicine-Rehab'
                 'Pediatrics-Neonatal' 'PainManagement' 'Orthopedic' 'Ophthalmology'
                 'OfficeNotes' 'Obstetrics/Gynecology' 'Neurosurgery' 'Nephrology'
                 'Letters' 'LabMedicine-Pathology' 'IME-QME-WorkCompetc.'
                 'Hospice-PalliativeCare' 'Hematology-Oncology' 'Gastroenterology'
                 'ENT-Otolaryngology' 'Endocrinology' 'EmergencyRoomReports'
                 'DischargeSummary' 'DietsandNutritions' 'Dermatology'
                 'Cosmetic/PlasticSurgery' 'Consult-HistoryandPhy.' 'Chiropractic'
                 'Autopsy']
 
So, to get a better understanding of our dataset, I created the following barplot, in order of most to least, to visualize the distribution of transcription counts for each class.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img1.png)

Evidently, the figure shows very high class imbalance, ranging from 6 to 1088 transcripts. Surgery sticks out immediately having a much higher transcription count than the rest - the range of values will be problematic for classification. So, let's filter out classes that are unusable due to having low counts and those that are beyond the scope of this project. 

After examining Surgery class, it appears that it is composed of a mix of generalized and specialized procedures from other classes, along with other types of procedures that do not quite belong in the other 39 classes. So, for reasons of potential overlapping with others specialties, in addition to having vastly far more counts than the rest, I decided that it should be dropped, in spite of the data loss.

* For another project, or even go deeper into this one, I believe Surgery can be partitioned by procedural type and reallocated to its most appropriate specialty class, althought it will be time consuming to parse through 1000+ transcripts.

Moving along, I will filter out 'ambulatory' medical specialty classes such as 'OfficeNotes', 'Letters', 'IME-QME-WorkCompetc., and, 'Consult-HistoryandPhy'.
And, lastly, a cut-off threshold needs to be defined for the lower value classes. I've attempted to go a bit further using different thresholds (25 and 50 transcripts) along with using four different types of resampling methods (SMOTE, ADASYN, SMOTEEN, SMOTETOMEK); however, since we will be spliting the sets for training and validation, the count value reduces even further, and also the resampling methods only added more noise, perhaps because there is too few samples to synthesize meaningful replacements. Therefore, any classes under 75 counts were dropped, and, in total, 13 unique classes remained. The following barplot shows a much better balance between classes.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img2.png)

## Text Preprocessing

For our 13 classes, text preprocessing methods need to be implemented. The following methods begins with splitting transcriptions into lists of words, followed by lowercasing, expanding contractions (i.e., that's -> that is), removing punctuations, stopwords, digits, special characters, and outliers that I had picked up along the way ('mmddyyyy','abc', etc.). Furthermore, texts/tokens will be lemmatized, stemmed, and removed if text containts less than 3 characters. 

For evaluation and simple curiousity, I developed two preprocessing functions that perform all the above except the second function includes one more step of employing scispaCy's biomedical pipeline package for NLP, 'en_core_sci_sm'. This package contains over 100,000 biomedical vocabulary so it will be able to identify certain biomedical jargon that may go unnoticed without it. You can look more into it by checking out https://pypi.org/project/scispacy/. It contains more features such as abbreviation detection, genetic biomarkers, and details for every entity this package identifies.

### Text Data Feature Extraction - Count Vectorization (CV) and term frequency inverse document frequency (Tf-IDF)(TF)

So, before the fun part, I performed sklearn's 'train_test_split' function to get a 20% testing set and 80% training set. Reasons being that we do not want to fit and tranform our entire dataset with CV and TFIDF cause this will allow some validation information to leak into our classification process, and, which by doing so, will inflate our metrics, which would be great if it were real, but it's not - so, split first.

Count Vectorization and Tf-Idf were set to extract unigrams, bigrams and trigrams, max features were set to 5000 features; although, increasing number of features would lead to better classification rates, by doing so, requires higher computing power and memory.

Max_df was set to 95%, which eliminates any words that appeared upto 95% amount of transcriptions, thereby, removing overly common words. Min_df was set to 4 and 5 document appearances, tf-idf and cv, respectively, meaning that texts need to have appeared in 4 and 5 transcripts atleast.

For both methods, only training sets were fitted, followed by transformation of both training and testing sets. 

### Truncated SVD 

Truncated SVD was applied to text count/tf-idf sparse matrices - a conventional process known as latent semanatic analysis (LSA). The benefit of LSA is very rewarding, it transforms the term/sparse matrices into a "semantic" space of lower dimensionality and uses "K"-number of singular value decomposition to find meaning in hidden or latent semantics that are manifest in texts and docs; moreover, the proportional variance explained by the term sparse matrices can be explained with reduced number of features! See below.

The number of features reduced from 5000 to 775 for CV training set, and 5000 to 1123 for TF-IDF training set. 


                Estimating number of components need to explain 95.0% for CV...
                **************************************************
                775 components needed to explain 95.0% variance.
                **************************************************
                Initial Feature Training Set -->{(1968, 5000)}
                Initial Feature Testing Set -->{(492, 5000)}
                **************************************************
                Modified Feature Training Set -->(1968, 775)
                Modified Feature Testing Set -->(492, 775)
                **************************************************
                Explained proportional variance: 94%
                

                Estimating number of components need to explain 95.0% for TF-IDF...
                **************************************************
                Initial Feature Training Set -->{(1968, 5000)}
                Initial Feature Testing Set -->{(492, 5000)}
                **************************************************
                Modified Feature Training Set -->(1968, 1123)
                Modified Feature Testing Set -->(492, 1123)
                **************************************************
                Explained proportional variance: 94%

        
However, while the output matrices of CV and TF-IDF are normalized, the results of LSA are not, so normalization was embedded within the customized TSVD function to normalize results. 

The hurdle of performing TSVD is to find the appropriate number of estimators to explain the desired proportional variance. 
So, an ambulatory TSVD function was developed to calculcate the necessary amount of estimators that can explain 95% or any other % proportional variance.

## EDA and t-SNE 
Firstly, the mean number of preprocessed words were calculated for each class. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img3.png)

*
From the barplot, it can be argued that surgically/proecdurally based classess, Orthopedics and Neurosurgery, have higher text averages per transcription,
which may be due to the preciseness, detailed nature of documenting more complex procedures. In contrast, Radiologist have the least amount which may be
due to their field that vears towards a need for more succint and forthright explanations/diagnoses from imaging scans and MRIs. 

* Ironically, having shadowing experiences among these three fields, I can to the very least corroborate some of these speculations.

The proportion of the total number of preprocessed texts by class were calculated - shown below.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img4.png)

From this pie chart, orthopedics has the absolute highest, which is rationable, since it also ranked the highest in average texts per transcript, and Cardiovascular/Pulmonary is also
high in percentage, mainly due to having most transcripts in the training set. But, the main take away is the proportion of preprocessed texts are not too unfairly imbalanced.

Word Clouds!
These terms represent the top 50 most common texts, and unsurprisingly, 'patient, left, right, diagnosis, blood' and etc are most common. The type of terms that you would often associate with the medical field.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img5.png)

However, this does not signify having higher weight over others words. By contrast, the IDF, or inverse document frequency, is employed to lower the weights of common terms,
such as these. So, while they appear the most commonly, by the rules of IDF, their weighting drop due to this nature. IDF and TF, term frequency, combine mathematically to identify
words that hold value and meaningful to their corresponding class. By getting the idf from the TF-IDF vector (vector.idf_) attribute, you can see that the same word cloud words
are ranked lowest in idf. 


![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img14.png)

This distribution plot of IDF values is a good representation of where the terms scale thru Tf-Idf's perspective.

## Chi-Squared Tests - Correlated Terms by Class
Chi-squared tests were conducted to show highest correlated texts for every class and examine the discrepancies between countvectorized and tfidf terms.

==================================================================================================================================
        ==> Cardiovascular/Pulmonary:

          * Most Correlated Tf-Idf Texts: 
        coronary, arteri, coronary arteri, circumflex, atrial

          * Most Correlated CountVectorized Texts: 
        coronary, arteri, pulmonary, coronary arteri, artery
==================================================================================================================================
        ==> Neurology:

          * Most Correlated Tf-Idf Texts: 
        temporal, seizur, eeg, brain, gait

          * Most Correlated CountVectorized Texts: 
        brain, temporal, gait, motor, seizur
==================================================================================================================================
        ==> Urology:

          * Most Correlated Tf-Idf Texts: 
        peni, bladder, scrotal, foreskin, testi

          * Most Correlated CountVectorized Texts: 
        bladder, testi, peni, prostate, scrotal
==================================================================================================================================
        ==> GeneralMedicine:
          * Most Correlated Tf-Idf Texts: 
        neg, subject, sugar, rate rhythm murmur, heent
        
          * Most Correlated CountVectorized Texts: 
        neg, histori, heent, auscult, procedur
==================================================================================================================================
        ==> Radiology:

          * Most Correlated Tf-Idf Texts: 
        imag, exam, signal, myoview, adenosin

          * Most Correlated CountVectorized Texts: 
        patient, imag, histori, exam, incis
==================================================================================================================================
        ==> Orthopedic:

          * Most Correlated Tf-Idf Texts: 
        knee, screw, tourniquet, tendon, carpal

          * Most Correlated CountVectorized Texts: 
        knee, screw, medial, joint, tendon
==================================================================================================================================
        ==> Obstetrics/Gynecology:

          * Most Correlated Tf-Idf Texts: 
        uteru, cervix, fetal, uterine, vaginal

          * Most Correlated CountVectorized Texts: 
        uteru, cervix, uterine, fetal, vaginal
==================================================================================================================================
        ==> Nephrology:

          * Most Correlated Tf-Idf Texts: 
        renal, kidney, transplant, renal mass, renal diseas

          * Most Correlated CountVectorized Texts: 
        renal, kidney, renal mass, transplant, renal diseas
==================================================================================================================================
        ==> Hematology-Oncology:

          * Most Correlated Tf-Idf Texts: 
        lymphoma, basal cell carcinoma, basal cell, cell, chemotherapi

          * Most Correlated CountVectorized Texts: 
        cell, carcinoma, basal cell, basal cell carcinoma, lymphoma
==================================================================================================================================
        ==> Gastroenterology:

          * Most Correlated Tf-Idf Texts: 
        colon, cecum, colonoscopi, scope, duodenum

          * Most Correlated CountVectorized Texts: 
        colon, gallbladd, duct, cecum, colonoscopi
==================================================================================================================================
        ==> ENT-Otolaryngology:

          * Most Correlated Tf-Idf Texts: 
        tonsil, nasal, ear, adenoid, media

          * Most Correlated CountVectorized Texts: 
        ear, nasal, tonsil, adenoid, pal


The main take away form the chi2 test for term correlations are that they both feature extraction methods do not vary much; however the results are a great indicator or relevent terms and attributes for each medical class.


Lastly, before modeling, we will take a look at LSA Tf-Idf and CV training values thru t-SNE scatterplots produced with cosine metric.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img6.png)



![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img7.png)




The visualizations provide very interesting insight into the allocation of class values on a 2-D plot. 


Both tf-idf and cv plots indicate positive and negative signs.
TF-IDF, aside from General Medicine which seems to be at the epicenter of slight overlap from nearly all classes, has many clusters that are present;
However, there are a few clusters that are concerning such as orthopedics/neurosurgery and neurology/radiology, these classes are tightly wound together in some areas.
This will prove to be difficult for classifying.
CountVectorization is also in a similiar state as tf-idf with the two set of clusters, orthopedics/neurosurgery and neurology/radiology. But it is clearly evident that all medical classes have some interdiscplinary overlap with general medicine.
Results will be interesting.












## Baseline Modeling
Get baseline models for the following seven classifiers:
*Stochastic Gradient Descent Classifier
*Multinomial Logistic Regression
*Logistic Regression OVR
*AdaBoosted Decision Trees Classifier
*Linear Support Vector Classifier OVR
*K-Nearest Neighbors Classifier
*LightGBM Classifier 

After analyzing preprocessed texts and visualizing TSNE scatterplots, I decided to use a variation of classifers; 
classifiers using OVR (OnevsRest, OVA, OVAOnevsAll), 
boosting, 
Multinomial Logistic Regression
I believe KNN Clf will have issues seperating clusters even with hyperparameter tuning, but I am very curious about how its' alghorithm will adjust.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img8.png)
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img9.png)
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img10.png)





The scispaCy model did show higher metrics compared to non-scispaCy metrics. And, considering the other functionalities the biomed sciscpaCy package contians, I assumme it is possible to improve metrics even further; for example, I did not convert medical abbreviations into words (MI = myocardialinfarction, BMI=bodymassindex, etc).

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
## Baseline Results 



## Hyperparameter Tuning using GridSearchCV and RandomSearchCV with 5-fold stratified cross-validaiton

RandomSearchCV is much faster than brute force GridSearchCV for tuning hyperparameters. Therefore using RandomSearchCV to narrow down optimal hyperparameters, I utilized GridSearchCV to confirm.

For LightGBM, I used Optuna, a hyperparameter optimization framework, for finding optimal hyperparameters at learning rate of 0.01. It calculates number of estimators, feature_fraction, num_leaves,  To avoid overfitting, the final step employs several regularization methods to best adjust the multi_logloss. 

## Final Results
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img11.png)
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img12.png)
Stochastic Gradient Descent Classifier with Tf-Idf vectorization produced highest F1-score, %. 

Logistic Regression OVR 

Multinomial Logistic Regression came close at second best score, %.



Tf-Idf produced higher F1-scores 

Linear Support Vector Classifier: Countvectorized tuned model outperformed Tf-Idf models.

LightGBM Classifier: Tfidf tuned model produced highest
AdaBoostedTree Classifier: Benefited greatly from hyperparameter tuning and produced a F1-score of 
## Best Estimator - Stoachstic Gradident Descent - Tf-IDF/LSA
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img13.png)
## Conclusions
Including scispaCy's biomed package helped improve metrics for all classifiers with the exception of KKN Clf. It would be interesting to look deeper into what other funcionalities it possess that can possibly improve classification rates even more. Initially, I hypothesized GeneralMedicine or Cardiovascular/pulmonary class to be the biggest obstacle for this project; since the nature of general medicine being more generalized would encompass factors that would overlap with other specialties and Cardiovascular/pulmonary, due to having the highest transcript count, as well as it being a medical field that has numerous associations with morbidities in other specilties. KNN clf, as I thought, was relatively consistent even without tuning, it was on par with other tuned models, since specialties would assumed to cluster as long as feature extraction worked properly; but there is an extent to KNN clf's capability to distinguish overlapping classes. As all classifiers did with radiology and neurology. Even SGD clf, the highest performing classifer, was only able to achieve an F1-score of 33%, but, Neurology did increase signficantly to 59%. 

Neurology and Radiology both comprise of similiar procedures such as MRIs and CT scans, in addition to neuroradiology, a specialty that wasn't accounted for with this dataset.

In conclusion, with these findings, I believe by implementing more text preprocessing methods such as conversion of medical jargon, acronyms, partitioning transcripts by common headers such as, "SUBJECTIVE, HISTORY, CHIEF COMPLAINT", and classifying based on these subsets; moreover, inclusion of more classes or breakdown of more class to have classes like "neuroradiology, interventional radiology, etc". And, of course, always, more data. 


## limitations
However, I have not attempted downsampling  majority classess, simply due to having already dropped a great amount of data, but it may be possible to partition and reallocate 'surgery' data by identifying specialty-based procedures. 

dropping
