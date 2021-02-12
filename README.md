# Text Classification of Medical Specialties and Evaluation of Different Preprocessing Methods using CountVectorization, TF-IDF, scispaCy's Biomedical NLP Package and Methodology Assessment

* Dataset: de-identified public source medical data from mtsamples.com
* Dataset details: contains 500 rows of de-identified medical information and five columns: medical specialty, descriptions, sample names, keywords

## Intro
Brief intro on TF-IDF, CountVectorization:

For algorithms to process language, language data, in any format (audio, documents, etc), needs to ultimately be converted into numerical vectors. Therefore, we rely on algorithms that can perform this conversion. For our purposes, we will use two vectorization techniques, CountVectorization and TF-IDF (term frequency inverse document frequency).

*hash vectorization is also available

Both vectorization methods extract features from corpus, that are composed of documents, and through conversion they return a term/document frequency sparse matrix. 

The main distinction between the two methods is IDF weighting. CountVectorization relies on count/frequency of terms to determine weights, while TF-IDF, is the term frequency multiplied by IDF weights. IDF or inverse document frequency, will penalize term weights based on how often they appear among total documents. We will see examples later below.

* from wiki, "tf–idf value increases proportionally to the number of times a word appears 
in the document and is offset by the number of documents in the corpus that contain the word".

## Objective: Classify medical specialties based on transcription notes using a combination of preprocessing methods. Achieveing highest F1-score is the aim.

We will be using the following seven classifiers.

### Classification Algorithms:
* Stochastic Gradient Descent Classifier OVR
* Multinomial Logistic Regression 
* Logistic Regression OVR
* AdaBoosted Decision Trees Classifier
* Linear Support Vector Classifier OVR
* K-Nearest Neighbors Classifier
* LightGBM Classifier 


### Approach
1. Split dataset into training and testing sets (test size @ 20%)

2. Build two types of text preprocessing functions:

    a. Lowercase, remove punctuations/stopwords/specialchars/digits/whitespace/terms with less than 2 chars, lemmatize and stem words/

    b. Second function will do the same, but will include scispaCy's en_core_sm biomed NLP package to assist in extracting biomedical term features.
 
3. Feature extraction of texts using CountVectorization and TF-IDF. Both parameters set at max 5000 features, inclusion of uni/bi/trigrams. Remove terms that occur within 95% of our transcriptions and a minimum threshold which discards terms that are not present in atleast 4,5 docs/transcripts for tf-idf and CV, respectively.


4. Apply Truncated SVD onto ONLY training sets then transform test and train sets using the fitted vector - otherwise we will have a case of data leakage and inflated metrics. Set a proportional variance percentage goal (95% will be used) you would like to have explained of the original . The sequential process of vectorization and TSVD is also known as LSA, latent semantic analysis.

### Import dataset and packages

After importing dataset, we extract our two columns of interest, transcription and medical specialty. Since, we only have 0.66% of missing values, we will remove them. Let's take a look at our target variable, medical specialty.

Target variable consists of 40 unique classes:

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
 
So, we see that there are several classes that are not releveant for our objective such as OfficeNotes, Letters, SOAP charts. We will remove them, but before doing so let's examine the distribution of transcriptions for our classses.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img1.png)


Clearly our classes are imbalanced; classes range from from 6 to 1088 transcripts, which is problematic. In addition, we can see that Surgery has far more values than the others by a significant amount. Lastly, we see that a handful of our classes have very few transcriptions which we will unfortunately have to account for by screening them out.

So after diving deeper into Surgery, it appears that it is composed of a mix of generalized and specialized procedures from other classess. It is concerning that Surgery contains information that overlap with many of our specialties, in addition to how it is imbalacing our data. Therefore we will drop it, in spite of the lost data.

* Note: Surgery data can be salvaged, perhaps, through MeSH techniques

Next, filter out our secondary classes, 'OfficeNotes', 'SOAP/Chart/ProgressNotes', 'Letters', 'IME-QME-WorkCompetc.', 'DischargeSummary' and, 'Consult-HistoryandPhy'.

Now we need to decide on a threshold cut-off for lower classes. So, I've went a bit further along attempting different thresholds (25 and 50 transcripts), along with attempting four types of resampling methods (SMOTE, ADASYN, SMOTEEN, SMOTETOMEK) - all of which actually decreased performance, probably, in result of oversampling from a few transcriptions - high bias. Therefore, we will set our cut-off threshold at 75 and above. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img2.png)

With our 13 classes, our distribution balance significantly improved.

## Text Preprocessing

To get better classification results, documents need to be cleansed and extracted og important words that help define its' class. As mentioned above, we will splitting our transcripts into bag of words, lowercasing all words, removing punctuations, stopwords, digits, special characters, and outliers that I had picked up along the way ('mmddyyyy','abc', etc.). And, lastly, lemmatizing and stemming words,

For evaluation of scispaCy's 'en_core_sm' biomedical NLP package, we will develop two preprocessing functions that perform all the above with one of the functions containing this package. 

This package contains over 100,000 biomedical vocabulary, contains more attributes and methods such as abbreviation detection and details on biomedical entities. Here is the url, https://pypi.org/project/scispacy/. 

### Text Data Feature Extraction - Count Vectorization (CV) and term frequency inverse document frequency (Tf-IDF)(TF)

Using the function we built, we will employ sklearn's 'train_test_split' function to output 20% testing set and 80% training set.

Now we are setting our parameters for Count Vectorization and Tf-Idf. Setting both to extract unigrams, bigrams and trigrams, or (1,3 n-grams), max features were set to 5000; although, a higher number of features will allow for better classification scores, but this requires higher computing power and memory, which I do not currently have.

Max_df are set to 95%, meaning that terms that appear in 95% of our total number of transcriptions will be removed. Furthermore, Min_df are set to terms having to appear in atleast 4 and 5 document appearances for tf-idf and cv, respectively. We do not want terms that are too common or too rare. 

And finally, both methods were fit to only the training sets and then transformed both training and testing sets using the vector that was fitted on training set.

So now, we have converted our text strings into numerical vectors that represent our language! 

### LSA - Truncated SVD 

Truncated SVD, similiar but not equivalent to PCA, will be applied to our count/inverse document freq sparse matrices. The benefit of LSA is  it transforms our count/sparse matrices into a "semantic" space of lower dimensionality and uses "K"-number of singular value decomposition to find hidden, latent meanings within our texts wthat allowing us, or the computer, to understand the contexts, terms, etc with lower number of features.  

The number of features reduced from 5000 to 775 for our CV training set, and 5000 to 1123 for TF-IDF training set. I had set my function to try to estimate 95% proportional variance, however we got 94% which is very close and will gladly accept.


                Estimating number of components need to explain 95.0% for CV...
                **************************************************
                764 components needed to explain 95.0% variance.
                **************************************************
                Initial Feature Training Set -->{(1968, 5000)}
                Initial Feature Testing Set -->{(492, 5000)}
                **************************************************
                Modified Feature Training Set -->(1968, 764)
                Modified Feature Testing Set -->(492, 764)
                **************************************************
                Explained proportional variance: 94%
                

                Estimating number of components need to explain 95.0% for TF-IDF...
                **************************************************
                Initial Feature Training Set -->{(1968, 5000)}
                Initial Feature Testing Set -->{(492, 5000)}
                **************************************************
                Modified Feature Training Set -->(1968, 1126)
                Modified Feature Testing Set -->(492, 1126)
                **************************************************
                Explained proportional variance: 94%

        
Another note to mention is that while the output matrices of CV and TF-IDF are normalized, the results of TSVD/LSA are not, so normalization is required for our output. So, I embdedded normalization within our function.

Lastly, for performing TSVD we need to estimate the appropriate number of estimators that can explain our desired proportional variance. 
So, with the use of a helper function for calculating the amount of estimators needed, our TSVD function is ready to transform, normalize, and output  desired proportional variance (I selected 95%).

### EDA and t-SNE visualizaiton
Now that we have preprocessed our transcripts, I'd like to calculate the mean and total number of 'cleaned' words for each class. 
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img_19.PNG)

The distribution is fairly even. But it can be argued that surgically dominant specialties such as our first and second classess, Orthopedics and Neurosurgery, have higher text averages per transcription; perhaps, due to the preciseness and detailing nature demanded for documenting more complex procedures. While in contrast, Radiology has the least average amount which makes sense given their role; radiologists need to communicate succinctly and be forthright when examining and diagnosising imaging scans and MRIs. 


Proceedingly along, I thought it would be interesting to see the distribution of cleaned words.
The following pie chart provides an easy way to analyze this.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img3.png)

Once again, orthopedics ranks the highest, which is rationable, since it had the highest mean texts per transcript; followed by Cardiovascular/Pulmonary which is also has a high percentage, but again, not very surprising, since it has the most transcripts out of our 13 classes. But the pie chart is a good indicator that the proportion of clean texts are not too unfairly balanced.

Onto Word Clouds!
The following wordcloud displays the top 50 most common texts found in our training data, and again, unsurprisingly, well known medical jargon such as 'patient, left, right, diagnosis, blood' and etc are leading in count.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img5.png)


For contrast when comparing outputs of TF-IDF. To reiterate, simply having a high count will not deliver you high term weight, inverse document frequency needs to be accounted for - this is tf-idf gifts to us.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img_16.PNG)

                
You can see that the most common words, the largest words in the wordcloud, now appear to be in the bottom for both TF-IDF and IDF. That is the power of IDF weighting. The higher the idf or tfidf goes, more frequently bigrams and trigrams seem to appear.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img4.png)

The IDF distribution plot is slightly negatively skewed, and texts were annotated across ranging IDF values (every 50th term between 0 to 5000) demcarcated by their X position. The terms become more sophisticated as the IDF or X axis increases. The term 'patient', 'procedure patient', 'drug' are the lowest IDF terms on this plot just like the figure above.

Chi-squared Tests for Tf-idf and CV were performed compare the differences in correlated terms to their class. With the number of unqiue terms betweeen TfIDF and CVect.


==> GeneralMedicine:

  * Most Correlated Tf-Idf Texts: 
heent, throat, distress, blood sugar, suppl, sugar, tender, histori, moist, neck suppl, mucous membran, neg, auscult, sound, subject

  * Most Correlated CountVectorized Texts: 
procedur, heent, emergency, daili, blood pressur, sugar, suppl, breath, sleep, histori, neg, auscult, right, medic, sound

Num of Differences Between Tf-Idf and CV in top 15: 16


==> Radiology:

  * Most Correlated Tf-Idf Texts: 
tricuspid, myoview, contrast, valv, veloc, signal, impress, adenosin, exam, patient, simul, fetal, stress, perfusion, imag

  * Most Correlated CountVectorized Texts: 
motor, incis, signal, impress, vicryl, room, diagnosi, exam, histori, sutur, patient, anesthesia, procedur, imag, skin

Num of Differences Between Tf-Idf and CV in top 15: 20


==> Urology:

  * Most Correlated Tf-Idf Texts: 
glan, penile, bladder, cystoscopi, prostat, foreskin, peni, testicl, urethra, scrotal, scrotum, prostate, circumcis, inguinal, testi

  * Most Correlated CountVectorized Texts: 
glan, penile, urethral, cystoscopi, prostat, prostate, peni, bladder, testicl, urethra, scrotal, scrotum, bladder neck, inguinal, testi

Num of Differences Between Tf-Idf and CV in top 15: 4

==> Ophthalmology:

  * Most Correlated Tf-Idf Texts: 
phacoemulsif, scleral, right eye, lid, eye, chamb, speculum, len, cataract, anterior chamb, ey, limbu, intraocular, right ey, capsular bag

  * Most Correlated CountVectorized Texts: 
phacoemulsif, right eye, lid, eye, chamb, speculum, len, cataract, anterior chamb, ey, intraocular, corneal, right ey, eyelid, capsular bag

Num of Differences Between Tf-Idf and CV in top 15: 4

==> Obstetrics/Gynecology:

  * Most Correlated Tf-Idf Texts: 
fallopian, placenta, infant, cervix, vagina, fallopian tub, labor, vaginal, babi, pregnanc, uterine, ovari, fetal, uteru, deliveri

  * Most Correlated CountVectorized Texts: 
fallopian, placenta, infant, cervix, vagina, fallopian tub, pelvic, vaginal, uterine, pregnanc, uteru, ovari, fetal, deliveri, peritoneum

Num of Differences Between Tf-Idf and CV in top 15: 4

==> Nephrology:

  * Most Correlated Tf-Idf Texts: 
kidney, kidney diseas, stage renal diseas, transplant, end stage, end stage renal, diagnosi end, diagnosi end stage, chronic kidney, renal diseas, renal, stage renal, dialysi, renal mass, cephalic vein

  * Most Correlated CountVectorized Texts: 
kidney, stage renal diseas, transplant, renal mass, end stage, end stage renal, diagnosi end, diagnosi end stage, nephrostomy, renal diseas, renal, stage renal, dialysi, vein, cephalic vein

Num of Differences Between Tf-Idf and CV in top 15: 4

You can see that more specialized classes have less term differences between TFIDF and CV. But, General Medicine and Radiology which overlaps with many other specialties have higher number of term differences, and it also insinuates that these two specialties are harder
to define by words.

* t-SNE

Lastly, before we move onto modeling, t-SNE visualization plots provide a very nice and simple way to visualize high dimensional features - our specialties and data points.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img6.png)

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img8.png)

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img15.png)

You can see that TF-IDF compared to CV had more defined clusters which will make it easier to classify. CV also had clusters however had a larger area composed of a mix of classes. It is difficult to see significant difference between non scispaCy and scispaCy, other than that scispaCy have multiple clusters of the same class in several areas, while non scispaCy classes seem to be in one large cluster.

T-SNE was estimated using cosine metric.


## Baseline Modeling
Classification Algorithms:
* Stochastic Gradient Descent Classifier OVR
* Multinomial Logistic Regression  and OVR
* Logistic Regression OVR
* AdaBoosted Decision Trees Classifier
* Linear Support Vector Classifier OVR
* K-Nearest Neighbors Classifier
* LightGBM Classifier 
### TF-ID scispaCy Classifcation Reports



* SGDClassifier 

                ===========================================================================
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.69      0.54      0.61        74
                      ENT-Otolaryngology       0.83      0.79      0.81        19
                        Gastroenterology       0.67      0.73      0.70        45
                         GeneralMedicine       0.62      0.71      0.66        52
                     Hematology-Oncology       0.44      0.39      0.41        18
                              Nephrology       0.56      0.31      0.40        16
                               Neurology       0.27      0.22      0.24        45
                            Neurosurgery       0.22      0.21      0.22        19
                   Obstetrics/Gynecology       0.72      0.84      0.78        31
                           Ophthalmology       0.94      0.94      0.94        16
                              Orthopedic       0.60      0.69      0.64        71
                               Radiology       0.09      0.09      0.09        55
                                 Urology       0.75      0.87      0.81        31

                                accuracy                           0.55       492
                               macro avg       0.57      0.56      0.56       492
                            weighted avg       0.55      0.55      0.55       492
                ===========================================================================
                
* LogisticRegression_OvR 

                ***************************************************************************
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.66      0.77      0.71        74
                      ENT-Otolaryngology       0.92      0.63      0.75        19
                        Gastroenterology       0.82      0.69      0.75        45
                         GeneralMedicine       0.53      0.83      0.65        52
                     Hematology-Oncology       0.67      0.11      0.19        18
                              Nephrology       0.67      0.25      0.36        16
                               Neurology       0.55      0.51      0.53        45
                            Neurosurgery       0.36      0.21      0.27        19
                   Obstetrics/Gynecology       0.71      0.77      0.74        31
                           Ophthalmology       1.00      0.88      0.93        16
                              Orthopedic       0.65      0.86      0.74        71
                               Radiology       0.18      0.13      0.15        55
                                 Urology       0.84      0.84      0.84        31

                                accuracy                           0.63       492
                               macro avg       0.66      0.57      0.58       492
                            weighted avg       0.62      0.63      0.60       492
                ===========================================================================
                
* LogisticRegression_Multinomial 

                ***************************************************************************
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.65      0.70      0.68        74
                      ENT-Otolaryngology       0.92      0.63      0.75        19
                        Gastroenterology       0.78      0.71      0.74        45
                         GeneralMedicine       0.55      0.79      0.65        52
                     Hematology-Oncology       0.80      0.22      0.35        18
                              Nephrology       0.57      0.25      0.35        16
                               Neurology       0.52      0.49      0.51        45
                            Neurosurgery       0.31      0.21      0.25        19
                   Obstetrics/Gynecology       0.74      0.81      0.77        31
                           Ophthalmology       1.00      0.88      0.93        16
                              Orthopedic       0.65      0.83      0.73        71
                               Radiology       0.17      0.15      0.16        55
                                 Urology       0.84      0.84      0.84        31

                                accuracy                           0.62       492
                               macro avg       0.65      0.58      0.59       492
                            weighted avg       0.61      0.62      0.60       492
                ===========================================================================
                
* LinearSVC 

                ***************************************************************************
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.68      0.61      0.64        74
                      ENT-Otolaryngology       0.83      0.79      0.81        19
                        Gastroenterology       0.70      0.73      0.72        45
                         GeneralMedicine       0.64      0.73      0.68        52
                     Hematology-Oncology       0.46      0.33      0.39        18
                              Nephrology       0.45      0.31      0.37        16
                               Neurology       0.28      0.20      0.23        45
                            Neurosurgery       0.29      0.32      0.30        19
                   Obstetrics/Gynecology       0.71      0.81      0.76        31
                           Ophthalmology       0.94      0.94      0.94        16
                              Orthopedic       0.62      0.70      0.66        71
                               Radiology       0.07      0.07      0.07        55
                                 Urology       0.77      0.87      0.82        31

                                accuracy                           0.57       492
                               macro avg       0.57      0.57      0.57       492
                            weighted avg       0.56      0.57      0.56       492
                ===========================================================================
                
  * AdaBoostClassifier 
  
                ***************************************************************************
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.48      0.54      0.51        74
                      ENT-Otolaryngology       0.64      0.37      0.47        19
                        Gastroenterology       0.50      0.40      0.44        45
                         GeneralMedicine       0.42      0.73      0.53        52
                     Hematology-Oncology       0.38      0.17      0.23        18
                              Nephrology       0.12      0.06      0.08        16
                               Neurology       0.17      0.13      0.15        45
                            Neurosurgery       0.13      0.11      0.12        19
                   Obstetrics/Gynecology       0.64      0.52      0.57        31
                           Ophthalmology       0.93      0.81      0.87        16
                              Orthopedic       0.50      0.62      0.55        71
                               Radiology       0.08      0.07      0.08        55
                                 Urology       0.62      0.52      0.56        31

                                accuracy                           0.42       492
                               macro avg       0.43      0.39      0.40       492
                            weighted avg       0.41      0.42      0.41       492
                ===========================================================================
                
* KNeighborsClassifier 

                ***************************************************************************
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.62      0.69      0.65        74
                      ENT-Otolaryngology       0.94      0.79      0.86        19
                        Gastroenterology       0.76      0.76      0.76        45
                         GeneralMedicine       0.65      0.69      0.67        52
                     Hematology-Oncology       0.50      0.33      0.40        18
                              Nephrology       0.56      0.56      0.56        16
                               Neurology       0.40      0.42      0.41        45
                            Neurosurgery       0.31      0.26      0.29        19
                   Obstetrics/Gynecology       0.75      0.87      0.81        31
                           Ophthalmology       1.00      0.94      0.97        16
                              Orthopedic       0.61      0.75      0.67        71
                               Radiology       0.18      0.11      0.13        55
                                 Urology       0.77      0.77      0.77        31

                                accuracy                           0.61       492
                               macro avg       0.62      0.61      0.61       492
                            weighted avg       0.59      0.61      0.60       492
                ===========================================================================
                
* LGBMClassifier 

                ***************************************************************************
                Classification Report
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.67      0.64      0.65        74
                      ENT-Otolaryngology       0.83      0.79      0.81        19
                        Gastroenterology       0.69      0.60      0.64        45
                         GeneralMedicine       0.52      0.62      0.56        52
                     Hematology-Oncology       0.36      0.28      0.31        18
                              Nephrology       0.42      0.31      0.36        16
                               Neurology       0.18      0.16      0.17        45
                            Neurosurgery       0.21      0.21      0.21        19
                   Obstetrics/Gynecology       0.69      0.71      0.70        31
                           Ophthalmology       0.94      0.94      0.94        16
                              Orthopedic       0.59      0.68      0.63        71
                               Radiology       0.11      0.11      0.11        55
                                 Urology       0.79      0.84      0.81        31

                                accuracy                           0.53       492
                               macro avg       0.54      0.53      0.53       492
                            weighted avg       0.52      0.53      0.52       492
 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img11.png)

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img_17.PNG)

So, from our results it is quite evident that all three preprocessing methods are fairly even in scoring. Adaboosted Trees are lowest, but that is to be expected, since it uses slow learners to amplify its classification. Both logistic regression models produced the highest F1 scores; LR is always reliable. For the most case, LSA tends to produce higher metrics as well as spaCy, but it is not convincing. So we will have to tune hyperparameters to get a better estimate of the impact of preprocessing methods. And, going through the classification reports, it is clear that radiology, neurology, and neurosurgery are the most difficult to classify.

* Main issue: radiology and neurology/neurosurgery


## Hyperparameter Tuning using GridSearchCV and RandomSearchCV with 5-fold stratified cross-validaiton

For hyperparameter tuning, I combined two functions, RandomSearchCV and GridSearchCV. I used RandomSearch to narrow down hyperparameters and then feed the rest to a 5 fold GridSearch. RandomSearch is much faster than brute forcing GridSearchCV for tuning hyperparameters and more flexible, so I tried to take advantage of the benefits of both functions. 

For LightGBM, I used Optuna, which is a hyperparameter optimization framework, for finding optimal hyperparameters at a learning rate, 0.01, that I decided on. How optuna works is that it sequentially calculates number of estimators, feature_fraction, num_leaves, among other hyperparameters in set order, and leaves regularizing for the final step. For our case, since it is multi-class and not binary we set our loss metric, multi_logloss. 

## Final Results
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img12.png)
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img_18.PNG)

The plot is between LSA+scispaCy and LSA models, so in theory, it should display the most discrepency of scispaCy's impact on metrics. 
Stochastic Gradient Descent Classifier with Tf-Idf vectorization produced highest F1-score, 70%. And LSA_spaCy tends to out perform in most models aside from KNN and Linear SVC. It is also interesting that the two boosting algortihmns, LGBM Classifier and Adaboosted Trees, did much poorly with count vectorization and scispaCy combined.



![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img13.png)

This plot shows the comparisons between LSA scispaCy, LSA and their baseline counterparts. 


## Best Estimator - Stoachstic Gradident Descent - Tf-IDF/LSA

Final look into SGD TF-IDF LSA_scispaCy model.

* Base LSA Tf-Idf SGD Model Classification report:

                ===========================================================================
                                          precision    recall  f1-score   support

                Cardiovascular/Pulmonary       0.69      0.54      0.61        74
                      ENT-Otolaryngology       0.83      0.79      0.81        19
                        Gastroenterology       0.67      0.73      0.70        45
                         GeneralMedicine       0.62      0.71      0.66        52
                     Hematology-Oncology       0.44      0.39      0.41        18
                              Nephrology       0.56      0.31      0.40        16
                               Neurology       0.27      0.22      0.24        45
                            Neurosurgery       0.22      0.21      0.22        19
                   Obstetrics/Gynecology       0.72      0.84      0.78        31
                           Ophthalmology       0.94      0.94      0.94        16
                              Orthopedic       0.60      0.69      0.64        71
                               Radiology       0.09      0.09      0.09        55
                                 Urology       0.75      0.87      0.81        31

                                accuracy                           0.55       492
                               macro avg       0.57      0.56      0.56       492
                            weighted avg       0.55      0.55      0.55       492

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

                Cardiovascular/Pulmonary       0.78      0.66      0.72        74
                      ENT-Otolaryngology       0.89      0.89      0.89        19
                        Gastroenterology       0.86      0.69      0.77        45
                         GeneralMedicine       0.58      0.73      0.65        52
                     Hematology-Oncology       0.52      0.67      0.59        18
                              Nephrology       0.44      0.69      0.54        16
                               Neurology       0.60      0.64      0.62        45
                            Neurosurgery       0.48      0.79      0.60        19
                   Obstetrics/Gynecology       0.79      0.87      0.83        31
                           Ophthalmology       1.00      0.94      0.97        16
                              Orthopedic       0.78      0.79      0.78        71
                               Radiology       0.55      0.31      0.40        55
                                 Urology       0.87      0.84      0.85        31

                                accuracy                           0.70       492
                               macro avg       0.70      0.73      0.71       492
                            weighted avg       0.71      0.70      0.69       492
 


Other than the high F1-score, it is great to see that Radiology increased from 9% to 40%, Neurology increased from 24% to 62%, and neurosurgery from 22% to 60%. I would conclude that thru tuning SGD was really able to shine and maximize its' potential. 

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img14.png)

The confusion matrix displays count and F1-Score for that class. While, Radiology had lowest F1-score, the improvement from its base model is significant. While Neurology and Neurosurgery misclassifications were not as sparse, radiology is seemingly more difficult to distinguish as its own seperate entity.



## Conclusions
Including scispaCy's biomed package helped improve F1 scores for all classifiers with the exception of KKN Clf and Linear SVC. It would be interesting to look deeper into what other possiblities we can implement with the packages' more complex functions. Initially, I theorized GeneralMedicine and Cardiovascular/pulmonary class to be the biggest obstacle for this project; since the nature of general medicine being more generalized would encompass factors that overlap with other specialties and Cardiovascular/pulmonary, for it having the highest transcript count as well as it being a specialty that has numerous morbidity associations with other class diseases. All classifiers had difficulties in distinguishing and classifying radiology and neurology. Even SGD clf, the highest performing classifer, was only able to achieve an F1-score of 40% for Radiology, although, the increase from baseline for both Radiology and Neurology was significant.

In conclusion, I believe the reason why Radiology and Neurology had been scoring low for all classifiers is that neuroradiology, within in its' own right, is an estanlished subspecialty of radiology and is widely recognized. Therer are several branches of radiology that specialize and manifest in entirely different ways, such as neuroradiology, as I've mentioned, as well as interventional radiology. So I strongly believe that if we were able to obtain data for neuroradiology, the classification metrics would increase for all classifers significantly. However, these are just my opinions and theory, but the findings are salient and vears me towards that reasoning. 

I hope you found this project interesting and let me know if you have any feedback on ways I can further improve.
Thanks for reading!

