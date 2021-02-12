# Text Classification of Medical Specialties and Evaluation of Different Preprocessing Methods using CountVectorization, TF-IDF, scispaCy's Biomedical NLP Package and Methodology Assessment (editing from tuning)

* Dataset: de-identified public source medical data from mtsamples.com
* Dataset details: contains 500 rows of de-identified medical information and five columns: medical specialty, descriptions, sample names, keywords

## Intro
Brief intro on TF-IDF, CountVectorization:

For algorithms to process language, language data, in any format (audio, documents, etc), needs to ultimately be converted into numerical vectors. Therefore, we rely on algorithms that can perform this conversion. For our purposes, we will use two vectorization techniques, CountVectorization and TF-IDF (term frequency inverse document frequency).

*hash vectorization is also available

Both vectorization methods extract features from corpus, that are composed of documents, and through conversion they return a term/document frequency sparse matrix. 

The main distinction between the two methods is IDF weighting. CountVectorization relies on count/frequency of terms to determine weights, while TF-IDF, is the term frequency multiplied by IDF weights. IDF or inverse document frequency, will penalize term weights based on how often they appear among total documents. We will see examples later below.

*from wiki, "tf–idf value increases proportionally to the number of times a word appears 
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
Now that we have preprocessed our transcripts, I'd like to calculate the mean number of 'cleaned' words for each class. 

The distribution is fairly even. But it can be argued that surgically dominant specialties such as our first and second classess, Orthopedics and Neurosurgery, have higher text averages per transcription; perhaps, due to the preciseness and detailing nature demanded for documenting more complex procedures. While in contrast, Radiology has the least average amount which makes sense given their role; radiologists need to communicate succinctly and be forthright when examining and diagnosising imaging scans and MRIs. 


Proceedingly along, I thought it would be interesting to see the distribution of cleaned words.
The following pie chart provides an easy way to analyze this.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img3.png)

Once again, orthopedics ranks the highest, which is rationable, since it had the highest mean texts per transcript; followed by Cardiovascular/Pulmonary which is also has a high percentage, but again, not very surprising, since it has the most transcripts out of our 13 classes. But the pie chart is a good indicator that the proportion of clean texts are not too unfairly balanced.

Onto Word Clouds!
The following wordcloud displays the top 50 most common texts found in our training data, and again, unsurprisingly, well known medical jargon such as 'patient, left, right, diagnosis, blood' and etc are leading in count.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img5.png)

For contrast when comparing outputs of TF-IDF. To reiterate, simply having a high count will not deliver you high term weight, inverse document frequency needs to be accounted for - this is tf-idf gifts to us.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img4.png)

The IDF distribution plot is slightly negatively skewed, and texts were annotated across ranging IDF values (every 50th term between 0 to 5000) demcarcated by their X position. The terms become more sophisticated as the IDF and X axis position increases. The term 'patient', 'procedure patient', 'drug' are the lowest IDF terms on this plot and it also seems as the IDF increases, the more number of bigrams and trigrams are present.  


![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img15.png)

                
You can plainly see that the most common words, the largest words in the wordcloud, now appear to be in the lowest ranking for IDF. Having taken account to inverse document frequency, popular terms can vary quite dramatically such as in our case, which is a great example of the differences of the two vectorization methods. 


Chi-squared Tests for Tf-idf and CV were performed compare the differences in correlated terms to their class. With the number of u



===========================================================================
==> GeneralMedicine:

  * Most Correlated Tf-Idf Texts: 
heent, throat, distress, blood sugar, suppl, sugar, tender, histori, moist, neck suppl, mucous membran, neg, auscult, sound, subject

  * Most Correlated CountVectorized Texts: 
procedur, heent, emergency, daili, blood pressur, sugar, suppl, breath, sleep, histori, neg, auscult, right, medic, sound

Num of Differences Between Tf-Idf and CV in top 15: 16

===========================================================================
==> Radiology:

  * Most Correlated Tf-Idf Texts: 
tricuspid, myoview, contrast, valv, veloc, signal, impress, adenosin, exam, patient, simul, fetal, stress, perfusion, imag

  * Most Correlated CountVectorized Texts: 
motor, incis, signal, impress, vicryl, room, diagnosi, exam, histori, sutur, patient, anesthesia, procedur, imag, skin

Num of Differences Between Tf-Idf and CV in top 15: 20

===========================================================================
==> Urology:

  * Most Correlated Tf-Idf Texts: 
glan, penile, bladder, cystoscopi, prostat, foreskin, peni, testicl, urethra, scrotal, scrotum, prostate, circumcis, inguinal, testi

  * Most Correlated CountVectorized Texts: 
glan, penile, urethral, cystoscopi, prostat, prostate, peni, bladder, testicl, urethra, scrotal, scrotum, bladder neck, inguinal, testi

Num of Differences Between Tf-Idf and CV in top 15: 4

===========================================================================
==> Ophthalmology:

  * Most Correlated Tf-Idf Texts: 
phacoemulsif, scleral, right eye, lid, eye, chamb, speculum, len, cataract, anterior chamb, ey, limbu, intraocular, right ey, capsular bag

  * Most Correlated CountVectorized Texts: 
phacoemulsif, right eye, lid, eye, chamb, speculum, len, cataract, anterior chamb, ey, intraocular, corneal, right ey, eyelid, capsular bag

Num of Differences Between Tf-Idf and CV in top 15: 4

===========================================================================
==> Obstetrics/Gynecology:

  * Most Correlated Tf-Idf Texts: 
fallopian, placenta, infant, cervix, vagina, fallopian tub, labor, vaginal, babi, pregnanc, uterine, ovari, fetal, uteru, deliveri

  * Most Correlated CountVectorized Texts: 
fallopian, placenta, infant, cervix, vagina, fallopian tub, pelvic, vaginal, uterine, pregnanc, uteru, ovari, fetal, deliveri, peritoneum

Num of Differences Between Tf-Idf and CV in top 15: 4

===========================================================================
==> Nephrology:

  * Most Correlated Tf-Idf Texts: 
kidney, kidney diseas, stage renal diseas, transplant, end stage, end stage renal, diagnosi end, diagnosi end stage, chronic kidney, renal diseas, renal, stage renal, dialysi, renal mass, cephalic vein

  * Most Correlated CountVectorized Texts: 
kidney, stage renal diseas, transplant, renal mass, end stage, end stage renal, diagnosi end, diagnosi end stage, nephrostomy, renal diseas, renal, stage renal, dialysi, vein, cephalic vein

Num of Differences Between Tf-Idf and CV in top 15: 4

You can see that more specizlied cases had less term differences between TFIDF and CV. But, General Medicine and Radiology which overlaps with many other specialties had high number of term differences,

Lastly, before we move onto modeling, t-SNE visualization plots provide a very nice way to visualize our specialties and data points.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img6.png)

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img8.png)

You can see that TF-IDF compared to CV had more defined clusters which will make it easier to classify. CV also had clusters however had a laragge area composed of a mix of classes. T-SNE was estimated using cosine metric.


## Baseline Modeling
Classification Algorithms:
* Stochastic Gradient Descent Classifier OVR
* Multinomial Logistic Regression  and OVR
* Logistic Regression OVR
* AdaBoosted Decision Trees Classifier
* Linear Support Vector Classifier OVR
* K-Nearest Neighbors Classifier
* LightGBM Classifier 



![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img11.png)

So, from our results it is quite evident that all three preprocessing methods are fairly even in scoring. Adaboosted Trees are lowest, but that is to be expected, since it uses slow learners to amplify its classification ability. Both logistic regression models produced the highest F1 scores; LR is always reliable. For the most case, LSA tends to produce higher metrics as well as spaCy, but it is not convincing. So we will have to tune hyperparameters to get a better estimate of the impact of preprocessing methods. 

 
* Main issue: radiology and neurology/neurosurgery


## Hyperparameter Tuning using GridSearchCV and RandomSearchCV with 5-fold stratified cross-validaiton

For hyperparameter tuning, I combined two functions, RandomSearchCV and GridSearchCV. I used RandomSearch to narrow down hyperparameters and then feed the rest to a 5 fold GridSearch. RandomSearch is much faster than brute forcing GridSearchCV for tuning hyperparameters and more flexible, so I tried to take advantage of the benefits of both functions. 

For LightGBM, I used Optuna, which is a hyperparameter optimization framework, for finding optimal hyperparameters at a learning rate, 0.01, that I decided on. How optuna works is that it sequentially calculates number of estimators, feature_fraction, num_leaves, among other hyperparameters in set order, and leaves regularizing for the final step. For our case, since it is multi-class and not binary we set our loss metric, multi_logloss. 

## Final Results
![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img12.png)

The plot is between LSA+scispaCy and LSA models, so in theory, it should display the most discrepency of scispaCy's impact on metrics. 
Stochastic Gradient Descent Classifier with Tf-Idf vectorization produced highest F1-score, 69.1%. And LSA_spaCy tends to out perform in every model. 

Next plot shows comparisons of LSA+scispaCy and LSA. In reality, the LSA model doesn't show much differences with the firsplt against raw values.tf-idf LSA spaCy continues to remain dominant ins highest f1 sorees all around.

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img13.png)

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

![alt text](https://github.com/cspark2610/medical-transcription-classification-/blob/main/images/img14.png)

The confusion matrix displays count and F1-Score for that class. While, Radiology had lowest F1-score, the improvement from its base model is significant.







Since our dependent/target varaible consists of many classes, I'd like to implement OVR  classifers with a primary focus on OVR classifers, since we have a large number of classes; perhaps, OVR can mitigate this issue by collapsing the classification process by piecemeal; OVR, or also goes by OVA (one vs all), sets one class against the others classes as a unified class, thereby simplifying the problem into multiple binary classifications. It is a promising approach rather than attempting to classify all targets at once. Boosters were selected cause I hoped that in the case where there are strong overlaps between neighboring classes, the boosting ability will adjust weights accordingly from pre classified and misclassified cases - which would be optimal. KNN classifier, I included, not particularly for classification score purposes, but more of an interest into how complex or overfit nearest neighbor alghorithm can extend to within domains and datasets such as medical notes. 







## Conclusions
Including scispaCy's biomed package helped improve F1 scores for all classifiers with the exception of KKN Clf. It would be interesting to look deeper into what other funcionalities it possesses that can possibly be used to improve classification rates. Initially, I theorized GeneralMedicine and Cardiovascular/pulmonary class to be the biggest obstacle for this project; since the nature of general medicine being more generalized would encompass factors that  overlap with other specialties and Cardiovascular/pulmonary, for it having the highest transcript count as well as it being a specialty that has numerous morbidity associations with other class diseases. All classifiers had difficulties in distinguishing and classifying radiology and neurology. Even SGD clf, the highest performing classifer, was only able to achieve an F1-score of 33%, although, Neurology did increase significantly to 59%. 

In conclusion, I believe the reason why Radiology and Neurology had been scoring low for all classifiers is that neuroradiology, within in its' own right, is an estanlished subspecialty of radiology and is widely recognized by medical entitities. Moreover, radiologists are all not the same, there are several branches of radiology that specialize and manifest in entirely different ways, such as neuroradiology, as I've mentioned, as well as interventional radiology. These subspecialties are well established and utilized in many hospitals, so I strongly believe that if we were able to obtain data for a neuroradiology class, the F1 scores for all classifers would significantly improve. However, these are just my opinions and theory, but the findings are salient and vears me towards that direction. 

I hope you found some insight through this project and thank you for reading it.

## Limitations and future options
I have not attempted downsampling majority classess, which may have improved scores, simply due to having a great deal of data loss from dropping surgery.
But, I believe surgery can be partitioned by procedural type and reallocate the data to make good use of it. 

Moreover, I believe implementing a more sophisticated text preprocessing method by addition of conversion of medical jargon, acronyms, and partitioning transcripts by common headers such as, "SUBJECTIVE, HISTORY, CHIEF COMPLAINT", can lead to more insight and similiarities in data structure. And, lastly, but not the very least, more data is always welcome. 
