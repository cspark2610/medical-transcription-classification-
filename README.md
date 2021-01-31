# Text Classifcation using CountVectorization, TF-IDF, scispaCy and Methodology Assessment on Improving Medical Specialty Classification Thru Transcription Notes

* Dataset: de-identified public source medical data from mtsamples.com
* Dataset details: contains 500 rows of de-identified medical information and 40 unique medical specialty classes, and also includes descriptions, sample names, and keywords.

## Intro

Brief intro on TF-IDF and CountVectorization.
Both methods are use to extract features from text data by converting texts (words, docs, tokens, strings) into numerical 
vectors so it can be processed by alghoritms. The main distinction between the two methods are the IDF weighting propeorty of
TF-IDF. 

*from wiki, 'tf–idf value increases proportionally to the number of times a word appears 
in the document and is offset by the number of documents in the corpus that contain the word'.

In other words, a term that appears very often in documents/corpora such as 'the' or 'and' will increase
in value due to its frequency, however since 'the' and 'and' appear in a great amount of documents as well, they are
penalized for it, and their weight values are adjusted to lower levels, indicating their lack in value to 'explain' their classes' variance, and are not very useful in NLP techniques.

CountVectorization does not take IDF into account. Unique terms are collected and are paired with values
that correspond to their frequency. Therefore, terms like 'the' and 'and' will stand out in CV extraction. Unfortunately for them,
a pre-filtering process using stopwords (list of common words that have little to no value 'the and he she I am from' and etc.,) removal. So, unfortunately, they will be filtered out again and have no place in the world of NLP.

Having said, I lean towards TF-IDF for text feature extraction rather than CountVect. Even with stopwords removal, IDF weighting is simply a much more valuable attribute to have and can be used in most domains.


## Objective: Classify medical transcript notes into predefined medical specialties with extra ambulatory features and evaluate findings.
We will be using the following seven classifiers.

### Classification Algorithms:
* Stochastic Gradient Descent Classifier OVR
* Multinomial Logistic Regression  and OVR
* Logistic Regression OVR
* AdaBoosted Decision Trees Classifier
* Linear Support Vector Classifier OVR
* K-Nearest Neighbors Classifier
* LightGBM Classifier 

I selected classifers with a primary focus on OVR classifers, since we have a large number of classes; perhaps, OVR can mitigate this issue by collapsing the classification process by piecemeal; OVR, or also goes by OVA (one vs all), sets one class against the others classes as a unified class, thereby simplifying the problem into multiple binary classifications. It is a promising approach rather than attempting to classify all targets at once. Boosters were selected cause I hoped that in the case where there are strong overlaps between neighboring classes, the boosting ability will adjust weights accordingly from pre classified and misclassified cases - which would be optimal. KNN classifier, I included, not particularly for classification score purposes, but more of an interest into how complex or overfit nearest neighbor alghorithm can extend to within domains and datasets such as medical notes. 

### Approach
1. Begin by spliting dataset into training and testing sets (test size @ 20%)

2. Construct text preprocessing functions:

    a. Lowercase, remove punctuations/stopwords/special chars/digits/whitespace, lemmatize and stem words and remove any terms with less than 2 chars.

    b. Second function follows same procedures but includes scispaCy's en_core_sm biomed package to assist in extracting biomedical terminology.

    c. Input preprocessing function into CountVec and TF-IDF's 'preprocessing' param.
    
    d. Comapare classification reports later on.
 
3. Feature extraction of texts using CountVectorization and TF-IDF. Both parameters were set
at max 5000 features, inclusion of uni/bi/trigrams. Removes terms occuring upyo 95% of transcriptions as well as a a threshold for being present in 4,5 docs/transcripts for tf-idf and CV, respectively.


4. Apply Truncated SVD onto ONLY training sets then transform test and train sets using this fitted vector - otherwise enables data leakage and inflation of metrics. Set a proportional variance percentage you would like to have explained of original set by reduced features (from 'k' number of singular vector decomposition' - beyond scope to go into details). This is a commonly used combination of textfeature extraction with TSVD, it is known as LSA or latent semantic analsys.

### Import dataset and packages

Starting off, extract our two columns, transcription and medical specialy, and then check and remove missing values, which happened to only be 0.66%.
Our dependent or target variable consists of 40 unique classes:

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
 
The following barplot displays the distribution of transcription counts for each class from most to least. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img1.png)

Evidently, the figure shows very high class imbalance, ranging from 6 to 1088 transcripts. Surgery sticks out immediately, having a much higher count than the others which is a bit of an issue for classification; especially given the handful of classes with lower values. So, it is time to start cutting out classes.

We will filter out classes that are unusable from low counts and irrelevant for the scope of this project.

Firstly, after examining Surgery, it appears that it is composed of a mix of generalized and specialized procedures from other classes, along with other 'outlier' procedures that do not quite belong in the other 39 classes. For reasons of potentially strong overlap with others specialties, in addition to having vastly far more counts than the rest, I chose to drop this class, in spite of the data loss.

* Note: in another project, or even deeper into this one, I believe Surgery can be partitioned by procedural type and be reallocated to appropriate classes, although  parsing through 1000+ transcripts will be very time consuming.

Nexy, I filtered out secondary classes such as 'OfficeNotes', 'Letters', 'IME-QME-WorkCompetc., and, 'Consult-HistoryandPhy'.
And, lastly, we need to decide a suitable cut-off threshold that filters thru frequency counts amongst the lower classes. I've went a bit further attempting different thresholds (25 and 50 transcripts), along with using four different resampling methods (SMOTE, ADASYN, SMOTEEN, SMOTETOMEK), all of which actually lowered classifcation rate, but this may be a result of resampling from lower counts. 
Additionaly, with train,test,split our training set will be further reduced. Therefore, I decided that our threshold be set at 75 and above. This leads to having 13 unique classes. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img2.png)

It is great to see that our distribution is fairly balanced with a decent amount of data for all classes.

## Text Preprocessing

Before we beging feature extraction, a lot of work preprocessing needs to take place to streamline the vectorization process. This includes splitting our transcripts into lists of words, lowercasing all words, expanding contractions (i.e., that's -> that is), removing punctuations, stopwords, digits, special characters, and outliers that I had picked up along the way ('mmddyyyy','abc', etc.). Basically, any words or sequene of words you know does not help define the uniqueness of the class to allow for better classification. And, lastly, texts/tokens are lemmatized, stemmed, and terms with less tah n2 character counts will be discarded. 

For evaluation of scispaCy's 'en_core_sm' biomedical NLP package, I developed two preprocessing functions that perform all the above functions withe the exception of the latter function which will include scispaCy's functionality. 

You can look more into it. This package contains over 100,000 biomedical vocabulary, contains more attributes and methods such as abbreviation detection, genetic biomarkers, and details for biomedical entities. Here is the url, https://pypi.org/project/scispacy/. 

### Text Data Feature Extraction - Count Vectorization (CV) and term frequency inverse document frequency (Tf-IDF)(TF)

So, now we will intiate the feature extraction process. We will begin by performing sklearn's 'train_test_split' function to output 20% testing set and 80% training set. Reasons for why we are doing this first is because we do not want to fit and tranform our entire dataset with CV and TFIDF; by doing so, will allow some extra information regarding our validation set to leak into our classification process, which will inflate our metrics. So, split first.

Now we are setting our parameters for Count Vectorization and Tf-Idf. I set both to extract unigrams, bigrams and trigrams, or (1,3 n-grams), max features were set to 5000; although, a higher number of features will allow for better classification scores, but this will require higher computing power and memory, which I do not currently have at my disposal.

Max_df were set to 95%, meaning that terms that appear in 95% of our total number of transcriptions will be removed. Furthermore, Min_df was set to terms having to appear in atleast 4 and 5 document appearances for tf-idf and cv, respectively. We do not want terms that are too common or too rare. 

And finally, both methods were fit to only the training sets and then transformed both training and testing sets using the vector that was fitted on training set.
So now, we have converted our text strings into numerical vectors representening our language! The outputs are sparse matrices, which is perfect for the following method.

### Truncated SVD 

Truncated SVD, similiar but not equivalent to PCA, will be applied to our textcount/tf-idf sparse matrices - these two sequential processess are better known as latent semanatic analysis (LSA). Latent, meaning subtle or hidden; semantics, in the case of variable meanings of terms depending on context in which it is being used. The benefit of LSA is very rewarding, it transforms our term count/sparse matrices into a "semantic" space of lower dimensionality and uses "K"-number of singular value decomposition to find hidden, latent meanings and embeddings within our texts that allows us, or the computer, to understand the contexts, terms, etc of our data; and additionaly, by uncovering more conncetions between our texts, LSA allows our original data variance to be explained with reduced amount of features/data. As you can see below. 

The number of features reduced from 5000 to 775 for our CV training set, and 5000 to 1123 for TF-IDF training set. I had set my function to try to estimate 95% proportional variance, however we got 94% which is very close and will gladly accept.


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

        
Another note to mention is that while the output matrices of CV and TF-IDF are normalized, the results of TSVD/LSA are not, so normalization is required for our output. So to combat my laziness, the normalization function is embedded within the customized TSVD function, so will output normalized results. 

Lastly, the biggest hurdle, in my opinion, for performing TSVD is to find the appropriate number of estimators that can explain our desired proportional variance. 
So, with the help of a helper function for calculating the amount of estimators needed, our TSVD function is ready to transform, normalize, and output your desired proportional variance (I selected 95%).

### EDA and t-SNE visualizaiton
Now that we have preprocessed our transcripts, I'd like to calculate the mean number of 'cleaned' words for each class. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img3.png)

*
The distribution is fairly even. But it can be argued that surgically dominant specialties such as our first and second ranked classess, Orthopedics and Neurosurgery, have higher text averages per transcription; perhaps, be due to the preciseness and detailing nature required for documenting more complex procedures. And in contrast, Radiologist tend to have the least average amount which makes sense given their role in medicine. Radiologists need to communicate succinctly and be forthright when examining and diagnosising imaging scans and MRIs. 

* Fun fact: I've shadowed or worked amongst these three fields so as a layman, my speculations may hold the absolute least amount of merit, or maybe entirely biased! 

Proceedingly along, I thought it would be interesting to see the distribution of cleaned words.
The following pie chart provides an easy way to analyze this.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img4.png)

Once again, orthopedics ranks the highest, which is rationable, since it had the highest mean texts per transcript; followed by Cardiovascular/Pulmonary which is also
high in percentage, but again, unsurprising, since it has the most transcripts out of our 13 classes. But the pie chart is a good indicator that the proportion of clean texts are not too unfairly balanced.

Onto Word Clouds!
The following wordcloud displays the top 50 most common texts found in our training data, and again, unsurprisingly, known medical jargon such as 'patient, left, right, diagnosis, blood' and etc are it.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img5.png)

However, despite what you think, I didn't only construct the wordcloud for the enjoyment, but rather, it is useful for contrast when comparing outputs of TF-IDF. To reiterate, simply having a high count will not deliver you high term weight, document frequency needs to be accounted for - this is what tf-idf will gift to us.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img14.png)

The IDF distribution plot is negatively skewed, but not largely, and texts were annotated across ranging IDF values (every 250th term between 0 to 5000) demcarcated by their X position. It may not be obvious but the terms become more sophisticated as the IDF and X axis position increases. The term 'patient' and 'recovery room satisfactory' are the lowest and highest annotated text on the distribution plot and it is notable that bigrams and trigrams become more present as the IDF increases.

                  Features with lowest idf:
                  ===========================================================================
                  ['patient' 'right' 'left' 'year' 'procedur' 'diagnosi' 'histori' 'blood'
                   'pain' 'room' 'postop' 'posit' 'preoper' 'anesthesia' 'preoper diagnosi'
                   'skin' 'prep' 'drape' 'examin' 'postop diagnosi' 'condit' 'exam' 'normal'
                   'prep drape' 'day']

                  Features with highest idf:
                  ===========================================================================
                  ['huntington' 'superfici femoral arteri' 'medial malleolu' 'limit lesion'
                   'colostomi' 'neural foraminal stenosi' 'fenestr' 'fistulogram'
                   'superfici femoral' 'glenn']

                  Features with lowest tfidf:
                  ===========================================================================
                  ['vascular statu' 'foot prep' 'foot prep drape' 'stabl vascular statu'
                   'stabl vascular' 'sign stabl vascular' 'superiorli inferiorli'
                   'retaining retractor' 'retaining' 'modified' 'stenosi right'
                   'muscl separ' 'procedur anesthesia' 'year caucasian femal' 'vesicouterin'
                   'left fallopian' 'constant' 'dissection' 'mention' 'muscl separ midlin'
                   'separ midlin' 'incis subcutaneous' 'general endotrach'
                   'incis subcutaneous tissu' 'incis level']

                  Features with highest tfidf:
                  ===========================================================================
                  ['arterial' 'biliary' 'reason visit' 'ica' 'capac' 'motor unit' 'ice'
                   'tonsil' 'angiogram' 'tremor' 'eclampsia' 'vestibular' 'cataract' 'sleep'
                   'temporal' 'fetal' 'reason' 'epicondyl' 'instil' 'seed' 'cholelithiasi'
                   'neg' 'suit' 'indic' 'subject']

You can plainly see that the most common words, the largest words in the wordcloud, now appear to be in the lowest ranking for IDF. Having taken account to inverse document frequency, popular terms can vary quite dramatically such as in our case, which is a great example of the differences of the two vectorization methods. 
High IDF terms such as Huntington's is a word that can significnatly help classify its' corresponding class (neurology?), something that the word 'patient' cannot do.

Now onto my personal favorite, t-SNE plots. It is a tremendously benefical way to visualize high dimensional data into 2-D scatter plots. I've tuned the perplexity, learning rate, and exaggeration to try to display the most interpretable plot. I was half way successful. For both plots, mostly tf-idf, there is clearly evidence of clusters which is a good indication for classifcation; however, a few clusters are a bit concerning, particularly, the clusters that contain more than one class embedded classes within a single cluster such as Orthopedics and Neurosurgery, and Radiology with several classes. The general epicenter of overlaps, where "general medicine" can be found, seems to be a challenge, since a little of every class has some value integrated within this area. I am hopeful that it will not crush our classification metrics. 


Final note, t-SNE was produced using cosine metric.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img6.png)



![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img7.png)


## Baseline Modeling
Now we will get our baseline metrics by training our seven models at default hyperparameters:
Classification Algorithms:
* Stochastic Gradient Descent Classifier OVR
* Multinomial Logistic Regression  and OVR
* Logistic Regression OVR
* AdaBoosted Decision Trees Classifier
* Linear Support Vector Classifier OVR
* K-Nearest Neighbors Classifier
* LightGBM Classifier 



![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img8.png)
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img9.png)

So, from our results it is quite evident that all three preprocessing methods are fairly even in scoring. Adaboosted Trees are lowest, but that is to be expected, since it uses slow learners to amplify its classification ability. Both logistic regression models produced the highest F1 scores; I commend the robustness of LR. We can see that Tf-Idf tends to outperform CountVect in most baseline models. 
Yet still, there we be a bit of tuning to improve our scores. It is not obvious here but from a few classification reports for baseline classes, Radiology and Neurology seem to be the most problematic classes. I see it as an issue sinc etheir support levels indicate that both classes are fairly high, so I cannot assume the poor metrtics on sample size. But something has to be done to improve these scores.

        ===========================================================================
        Classification Report - SGDClassifier
        ===========================================================================
                                  precision    recall  f1-score   support

        Cardiovascular/Pulmonary       0.67      0.58      0.62        74
              ENT-Otolaryngology       0.83      0.79      0.81        19
                Gastroenterology       0.69      0.73      0.71        45
                 GeneralMedicine       0.58      0.63      0.61        52
             Hematology-Oncology       0.40      0.33      0.36        18
                      Nephrology       0.38      0.31      0.34        16
                       Neurology       0.26      0.22      0.24        45
                    Neurosurgery       0.26      0.26      0.26        19
           Obstetrics/Gynecology       0.69      0.81      0.75        31
                   Ophthalmology       0.94      0.94      0.94        16
                      Orthopedic       0.62      0.68      0.65        71
                       Radiology       0.07      0.07      0.07        55
                         Urology       0.76      0.84      0.80        31

                        accuracy                           0.54       492
                       macro avg       0.55      0.55      0.55       492
                    weighted avg       0.54      0.54      0.54       492

        ===========================================================================
        Classification Report - LogisticRegression_OvR
        ===========================================================================
                                  precision    recall  f1-score   support

        Cardiovascular/Pulmonary       0.63      0.74      0.68        74
              ENT-Otolaryngology       0.93      0.68      0.79        19
                Gastroenterology       0.78      0.64      0.71        45
                 GeneralMedicine       0.51      0.81      0.63        52
             Hematology-Oncology       0.50      0.06      0.10        18
                      Nephrology       0.67      0.25      0.36        16
                       Neurology       0.55      0.49      0.52        45
                    Neurosurgery       0.33      0.21      0.26        19
           Obstetrics/Gynecology       0.71      0.77      0.74        31
                   Ophthalmology       1.00      0.88      0.93        16
                      Orthopedic       0.65      0.85      0.74        71
                       Radiology       0.20      0.15      0.17        55
                         Urology       0.81      0.81      0.81        31

                        accuracy                           0.61       492
                       macro avg       0.64      0.56      0.57       492
                    weighted avg       0.60      0.61      0.59       492

        ===========================================================================
        Classification Report LogisticRegression_Multinomial 
        ===========================================================================
                                  precision    recall  f1-score   support
        Cardiovascular/Pulmonary       0.65      0.69      0.67        74
              ENT-Otolaryngology       0.93      0.68      0.79        19
                Gastroenterology       0.76      0.64      0.70        45
                 GeneralMedicine       0.51      0.77      0.61        52
             Hematology-Oncology       0.75      0.17      0.27        18
                      Nephrology       0.67      0.25      0.36        16
                       Neurology       0.53      0.47      0.49        45
                    Neurosurgery       0.38      0.32      0.34        19
           Obstetrics/Gynecology       0.71      0.77      0.74        31
                   Ophthalmology       1.00      0.88      0.93        16
                      Orthopedic       0.65      0.82      0.72        71
                       Radiology       0.19      0.16      0.17        55
                         Urology       0.84      0.84      0.84        31

                        accuracy                           0.61       492
                       macro avg       0.66      0.57      0.59       492
                    weighted avg       0.61      0.61      0.59       492

        ===========================================================================
        Classification Report LinearSVC
        ===========================================================================

                                  precision    recall  f1-score   support

        Cardiovascular/Pulmonary       0.68      0.61      0.64        74
              ENT-Otolaryngology       0.83      0.79      0.81        19
                Gastroenterology       0.73      0.73      0.73        45
                 GeneralMedicine       0.60      0.69      0.64        52
             Hematology-Oncology       0.43      0.33      0.38        18
                      Nephrology       0.42      0.31      0.36        16
                       Neurology       0.28      0.22      0.25        45
                    Neurosurgery       0.26      0.26      0.26        19
           Obstetrics/Gynecology       0.69      0.77      0.73        31
                   Ophthalmology       0.94      0.94      0.94        16
                      Orthopedic       0.61      0.68      0.64        71
                       Radiology       0.07      0.07      0.07        55
                         Urology       0.79      0.84      0.81        31

                        accuracy                           0.55       492
                       macro avg       0.56      0.56      0.56       492
                    weighted avg       0.55      0.55      0.55       492

        ===========================================================================
        Classification Report KNeighborsClassifier 
        ===========================================================================

                                  precision    recall  f1-score   support

        Cardiovascular/Pulmonary       0.59      0.72      0.65        74
              ENT-Otolaryngology       0.78      0.74      0.76        19
                Gastroenterology       0.78      0.69      0.73        45
                 GeneralMedicine       0.55      0.58      0.56        52
             Hematology-Oncology       0.47      0.39      0.42        18
                      Nephrology       0.50      0.44      0.47        16
                       Neurology       0.44      0.47      0.45        45
                    Neurosurgery       0.29      0.21      0.24        19
           Obstetrics/Gynecology       0.77      0.87      0.82        31
                   Ophthalmology       1.00      0.94      0.97        16
                      Orthopedic       0.63      0.73      0.68        71
                       Radiology       0.19      0.13      0.15        55
                         Urology       0.76      0.71      0.73        31

                        accuracy                           0.59       492
                       macro avg       0.59      0.58      0.59       492
                    weighted avg       0.57      0.59      0.58       492
 

  
* Main issue: radiology and neurology/neurosurgery (going back to t-SNE plot, this is expected; generalmedicine and cardiovascular/pulmonary were not as problematic    as I presumed they would be. 

## Hyperparameter Tuning using GridSearchCV and RandomSearchCV with 5-fold stratified cross-validaiton

For hyperparameter tuning, I combined the two functions, RandomSearchCV and GridSearchCV, and made a diction of all our classifers' hyperparameters. My approach that I used was to use RandomSearch to narrow down hyperparameters and then feed the rest to a 5 fold GridSearch. RandomSearch is much faster than brute forcing GridSearchCV for tuning hyperparameters and flexible, so I tend to combine the benefits of both to get the most out of our models. 

For LightGBM, I used Optuna, which is a hyperparameter optimization framework, for finding optimal hyperparameters at a learning rate, 0.01, that I decided on. How optuna works is that it sequentially calculates number of estimators, feature_fraction, num_leaves, among other hyperparameters in set order, and leaves regularizing for the final step. For our case, since it is multi-class and not binary we will set our loss metric, multi_logloss. 

## Final Results
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img12.png)

The plot is between LSA+scispaCy and LSA models, so in theory, it should display the most discrepency of scispaCy's impact on metrics. 
Stochastic Gradient Descent Classifier with Tf-Idf vectorization produced highest F1-score, 69.1%. And LSA_spaCy tends to out perform in every model. 

Next plot shows comparisons of LSA+scispaCy and LSA. In reality, the LSA model doesn't show much differences with the firsplt against raw values.tf-idf LSA spaCy continues to remain dominant ins highest f1 sorees all around.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img16.png)

This plot confirms that, in spite of which preprocessing method used for feature extraction, tf-idf produces higher scores on average than countvectorization.


Multinomial Logistic Regression came close at second with a final F1-score of 67% but SGD is definitively the best estimator.


## Best Estimator - Stoachstic Gradident Descent - Tf-IDF/LSA

Final look into SGD TF-IDF LSA_scispaCy model.

Base LSA Tf-Idf SGD Model Classification report:
===========================================================================
                          precision    recall  f1-score   support

Cardiovascular/Pulmonary       0.65      0.57      0.60        74
      ENT-Otolaryngology       0.73      0.84      0.78        19
        Gastroenterology       0.62      0.76      0.68        45
         GeneralMedicine       0.62      0.60      0.61        52
     Hematology-Oncology       0.44      0.39      0.41        18
              Nephrology       0.44      0.25      0.32        16
               Neurology       0.21      0.13      0.16        45
            Neurosurgery       0.27      0.32      0.29        19
   Obstetrics/Gynecology       0.69      0.81      0.75        31
           Ophthalmology       0.94      0.94      0.94        16
              Orthopedic       0.62      0.69      0.65        71
               Radiology       0.05      0.05      0.05        55
                 Urology       0.76      0.84      0.80        31

                accuracy                           0.54       492
               macro avg       0.54      0.55      0.54       492
            weighted avg       0.52      0.54      0.53       492

===========================================================================
Best Estimator - Stochastic Gradient Descent Classifier LSA Tf-IDF
===========================================================================
Parameters:
SGDClassifier(alpha=0.001, class_weight='balanced', eta0=0.01, l1_ratio=0.8,
              learning_rate='adaptive', loss='modified_huber',
              n_iter_no_change=10, n_jobs=-1, penalty='l1', power_t=0.01,
              random_state=123)

===========================================================================
Classification Report
===========================================================================
                          precision    recall  f1-score   support

Cardiovascular/Pulmonary       0.77      0.66      0.71        74
      ENT-Otolaryngology       0.85      0.89      0.87        19
        Gastroenterology       0.86      0.71      0.78        45
         GeneralMedicine       0.58      0.67      0.63        52
     Hematology-Oncology       0.48      0.67      0.56        18
              Nephrology       0.46      0.69      0.55        16
               Neurology       0.57      0.60      0.59        45
            Neurosurgery       0.48      0.84      0.62        19
   Obstetrics/Gynecology       0.77      0.87      0.82        31
           Ophthalmology       0.94      0.94      0.94        16
              Orthopedic       0.79      0.77      0.78        71
               Radiology       0.48      0.24      0.32        55
                 Urology       0.82      0.90      0.86        31

                accuracy                           0.68       492
               macro avg       0.68      0.73      0.69       492
            weighted avg       0.69      0.68      0.68       492
 

Other than the high F1-score, it is great to see that Radiology increased from 5% to 32% and Neurology increased from 16% to 59%. I would conclude that thru tuning SGD was really able to maximize its' potential. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img13.png)

The confusion matrix displays count and F1-Score for that class. While, Radiology had lowest F1-score, the improvement from its base model is significant.

## Conclusions
Including scispaCy's biomed package helped improve F1 scores for all classifiers with the exception of KKN Clf. It would be interesting to look deeper into what other funcionalities it possesses that can possibly be used to improve classification rates. Initially, I theorized GeneralMedicine and Cardiovascular/pulmonary class to be the biggest obstacle for this project; since the nature of general medicine being more generalized would encompass factors that  overlap with other specialties and Cardiovascular/pulmonary, for it having the highest transcript count as well as it being a specialty that has numerous morbidity associations with other class diseases. All classifiers had difficulties in distinguishing and classifying radiology and neurology. Even SGD clf, the highest performing classifer, was only able to achieve an F1-score of 33%, although, Neurology did increase significantly to 59%. 

In conclusion, I believe the reason why Radiology and Neurology had been scoring low for all classifiers is that neuroradiology, within in its' own right, is an estanlished subspecialty of radiology and is widely recognized by medical entitities. Moreover, radiologists are all not the same, there are several branches of radiology that specialize and manifest in entirely different ways, such as neuroradiology, as I've mentioned, as well as interventional radiology. These subspecialties are well established and utilized in many hospitals, so I strongly believe that if we were able to obtain data for a neuroradiology class, the F1 scores for all classifers would significantly improve. However, these are just my opinions and theory, but the findings are salient and vears me towards that direction. 

I hope you found some insight through this project and thank you for reading it.

## Limitations and future options
I have not attempted downsampling majority classess, which may have improved scores, simply due to having a great deal of data loss from dropping surgery.
But, I believe surgery can be partitioned by procedural type and reallocate the data to make good use of it. 

Moreover, I believe implementing a more sophisticated text preprocessing method by addition of conversion of medical jargon, acronyms, and partitioning transcripts by common headers such as, "SUBJECTIVE, HISTORY, CHIEF COMPLAINT", can lead to more insight and similiarities in data structure. And, lastly, but not the very least, more data is always welcome. 
