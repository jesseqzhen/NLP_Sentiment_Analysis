# Sentiment Analysis in Amazon Customer Reviews

###### |<a href='#1'> Introduction </a> | <a href='#2'> Data </a> | <a href='#3'> Preprocessing </a> |<a href='#4'> Feature Extraction </a>| <a href='#5'> Model </a> | <a href='#6'> Feature Selection </a> |<a href='#7'> Web App </a>| <a href='#8'> Discussion </a> |

###### |<a href='https://jesseqzhen.github.io/NLP_Sentiment_Analysis/'> Python Code HTML Version </a>| <a href='http://jesseqz.pythonanywhere.com/'> Web Application Website </a>|

<a id='1'></a>
## Introduction

One of the hot-trend topics in Natural Language Processing (NLP) is sentiment analysis. Sentiment analysis involves extraction of subjective information from documents like posts and reviews to determine the opinion with respect to products, service, events, or ideas.

This project uses the customer review data from Amazon.com to perform a supervised binary (positive or negative) sentiment classification analysis. We use various data pre-processing techniques and demonstrate their effectiveness in improving the classification. We also compare three machine learning models, namely, the multinomial Naive Bayes classification model (MultinomialNB), the Logistic regression model (LogisticRegression), and the linear support vector classification model (LinearSVC).  

The result of the analysis shows that adding negation handling and n-grams modeling techniques into data preprocessing can significantly increase the model accuracy. The result also indicates that SVC model provides the best prediction accuracy. 

<a id='2'></a>
## Data

#### Data Source
The data comes from the website ["Amazon product data"](http://jmcauley.ucsd.edu/data/amazon/) managed by Dr. Julian McAuley from UCSD. We choose the smaller subset of the customer review data from the Kindle store of Amazon.com [(link to download the dataset)](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz). The data is in the JSON format, which contains 982,619 reviews and metadata spanning May 1996 - July 2014. 

#### Sentiment Labeling
Reviews with overall rating of 1, 2, or 3 are labeled as negative ("neg"), and reviews with overall rating of 4 or 5 are labeled as positive ("pos"). Thus, number of positive and negative reviews are as follows in the original dataset:

* positive: 829,277 reviews (84.4%)
* negative: 153,342 reviews (15.6%)

#### Undersampling

Since the dataset is imbalanced that more than 84% of the reviews are positive, we undersample the positive reviews (the majority class) to have exactly the same number of reviews as in the negative ones.



<a id='3'></a>
## Preprocessing  

The following steps are used to preprocess data:

* Use HTMLParser to un-escape the text
* Change "can't" to "can not", and change "n't" to "not" (This is useful for the later negation handling process)
* Pad punctuations with blanks
  * Note: if choose not to perform negation handling, then remove punctuations
* Word normalization: lowercase every word
* Word tokenization
* Perform **negation handling**
  * A major problem faced during sentiment analysis is that of handling negations.
  * The algorithm used here comes from Narayanan, Arora, and Bhatia's paper "*Fast and accurate sentiment classification using an
enhanced Naive Bayes model*" [(link)](https://arxiv.org/abs/1305.6143)
  * The algorithm:
    * Use a state variable to store the negation state
    * Transform a word followed by a "not" or "no" into “not_” + word
    * Whenever the negation state variable is set, the words read are treated as “not_” + word
    * The state variable is reset when a punctuation mark is encountered or when there is double negation
* Use **bigram** or **trigram** model
  * Information about sentiment is often conveyed by adjectives ore more specifically by certain combinations of adjectives. 
  * This information can be captured by adding features like consecutive pairs of words (bigrams) or even triplets of words (trigrams).
* Word lemmatization

Then, we split the whole dataset randomly to the training set, validation set, and testing set by the proportion of 60%, 20%, and 20% respectively.
* Training set contains 184,010 reviews
* Validation set contains 61,337 reviews
* Testing set contains 61,337 reviews

<a id='4'></a>
## Feature Extraction

We use **vectorization** process to turn the collection of text documents into numerical feature vectors.
To extract numerical features from text content, we use the **Bag of Words** strategy:  

* **tokenizing** strings and giving an integer id for each possible token, by using white-spaces as token separators
* **counting** the occurrences of tokens in each document
* **normalizing** and **tf-idf weighting** with diminishing importance tokens that occur in the majority of documents

A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus, while completely ignoring the relative position information of the words in the document.

<a id='5'></a>
## Model

Next we demonstrate the effectiveness of negation handling and n-gram modeling techniques, and compare three machine learning algorithms, namely, the multinomial Naive Bayes classification model (MultinomialNB), the Logistic regression model (LogisticRegression), and the linear support vector classification model (LinearSVC).  

As a basic feature selection procedure, we remove features/tokens which occur only once to avoid over-fitting. We also use the default penalty parameter in each machine learning algorithm.  

The following table illustrates the model accuracy on the testing dataset by using different preprocessing procedures and different machine learning algorithms:

|Preprocessing procedure Added 	| Number of features/tokens | MultinomialNB  	| LogisticRegression  	| LinearSVC  	|
|---	                    |---	                    |---	            |---	                | ---           |
|Basic preprocessing^       | 56,558                    | 0.8329  	        | 0.8453   	            | 0.8485     	|
|Adding negation handling   | 71,853                    | 0.8262         	| 0.8519              	| 0.8562     	|
|Adding bigrams and trigrams| 2,027,753                 | 0.8584         	| 0.8675              	| 0.8731     	|

^ *Basic preprocessing* procedures include procedures with uni-gram modeling but without negation handling. 

The above table clearly shows that adding negation handling and n-grams modeling techniques can significantly increase the model accuracy. The table also indicates that SVC model provides the best prediction accuracy. 

<a id='6'></a>
## Feature Selection

The models trained in the above table come with a rather coarse feature selection procedure, that is, we simply remove features/tokens which occur only once.  

To reach a better prediction power, we can fine-tune the number of features needed for each algorithm by using the validation dataset. Here we perform all of the preprocessing procedures, including negation handling and bigrams/trigrams modeling. A plot of **Model Accuracy vs Number of Features** is shown below:

![table](model_accuracy.png?raw=true "Title")

It clearly shows that LinearSVC has the highest accuracy consistently, with LogisticRegression the less, and MultinomialNB the least. The following table summarizes the best number of features and the model accuracy on validation set and testing set.

|                          	| MultinomialNB  	| LogisticRegression  	| LinearSVC  	|
|---	                       |---	             |---	                  |---	         |
|Best number of features    | 1,000,000  	    | 500,000   	          | 1,700,000  	|
|Accuracy on validation set | 0.8580          | 0.8697              	| 0.8746     	|
|Accuracy on testing set    | 0.8585          | 0.8682              	| 0.8730     	|



<a id='7'></a>
## Web App

To demonstrate this project, we wrote a **flask** (a lightweight WSGI web application framework [[link]](http://flask.pocoo.org/)) 
web application. You can enter some review text and let the app analyze the sentiment of your entry. Have fun!

[NLP Sentiment Analysis App](http://jesseqz.pythonanywhere.com/)



<a id='8'></a>
## Discussion 
Here are some ending throughs through this project.

* We tried to remove stop words during the preprocessing stage but it decreased the model accuracy. We only used the stop words from nltk package, which may be the reason. A better way worth a try is to manually select stop words which are related to the current topic of the dataset. But since we have a very large dataset and we also perform tf-idf reweighting process, removing stop words may not be necessary. 

* To save tuning time, we did a pretty rough tuning on number of features with 100,000 number of features increment in each tuning. A finer tuning may get a better prediction power.

* We can also tune the penalty parameter in each machine learning algorithm. We currently only use the default penalty parameter. By tuning this parameter, we may get a better prediction power.

* We notice that under the current three machine learning models, parameter tuning may not provide significant accuracy boost. More advanced models such as Long short-term memory (LSTM) may be adopted. 


---

#### Reference

1. [Fast and accurate sentiment classification using an enhanced Naive Bayes model,](https://arxiv.org/abs/1305.6143) Intelligent Data Engineering and Automated Learning IDEAL 2013 Lecture Notes in Computer Science Volume 8206, 2013, pp 194-201.
