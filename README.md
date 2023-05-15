In this ML project I use an extensive dataset that contains three decades worth of Greek parliamentary proceedings.   

# The Dataset

[The dataset](https://zenodo.org/record/2587904) originated from the work implemented during the course of the Master thesis entitled "Speech quality and sentiment analysis on the Hellenic Parliament proceedings" at the Athens University of Economics & Business in 2018. It has been updated multiple times since then, in order for the best result to be achieved.   

It includes 1,194,407 speeches of Greek parliament members with a total volume of 2.15 GB, that where exported from 5,118 parliamentary sitting record files and extend chronologically from 1989 up to 2019.   

The dataset consists of a csv file in UTF-8 encoding and includes the following columns of data:    

**member_name**: The official name of the parliament member that talked during a sitting.    

**sitting_date**: The date that the sitting took place. There are cases were more than one sittings took place at the same date.     

**parliamentary_period**: The name and/or number of the parliamentary period that the speech took place in. A parliamentary period includes multiple parliamentary sessions.   

**parliamentary_session**: The name and/or number of the parliamentary session that the speech took place in. A parliamentary session includes multiple parliamentary sittings.   

**parliamentary_sitting**: The name and/or number of the parliamentary sitting that the speech took place in.     

**political_party**: The political party that the speaker belongs to.   

**speaker_info**: Information about the speaker extracted from the text of the proceeding/sitting record that refers to the parliamentary role of the speaker such as Chairman of the Parliament, Finance Minister or similar.     

**speech**: The speech that the member made during the sitting of the Greek Parliament.   

My final **goal** is to perform **classification** (speech --> political party) with and without the use of Neural Networks.    
 
To achieve this goal I used **Python Pandas**, **Jupyter Notebook** and followed the below process:   

# PART 1 - Data Exploration and Trasformation 

In this part I explore the dataset to get an overall idea of what it is about.   
More specifically I extract **metrics** such as:   

    - Number of different political parties
    - Speeches per political party
    - Number of different speakers per political party
    - Number of speeches per political party per year 
    - Length of each speech (in words)
    - Number of unique words used for each speech length
    - Frequency of each word by using nltk library

I also **visualize** the above metrics using the libraries **seaborn** and **matplotlib**.    

In Part 1 I also proceed to clean and transform the dataset to be ready for the next step (classsification).    
More specifically I:   

    - Delete rows that have NaN values
    - Delete entries from political parties that are underrepresented in the dataset (less that 1000 speeches)
    - Add a new column that represents the length of each speech
    - Categorize speeches to bins depending on speech length
    - Delete speeches that have less than 15 OR more than 1000 words
    - Remove punctuations and stop words (very frequent and not at all frequent words) from speeches
    - lemmatize words using spacy (nlp library that supports Greek language)


# PART 2 - Machine Learning (classification without NNs)

In this part I train 3 **non-neural network** algorithms to classify speeches.   
The **target variable** is the political party of the speaker.   

I noticed that the dataset is pretty **imbalanced** so I proceeded to **upsample** some political parties using **sklearn**.   

I split the Data into **Train** and **Test** datasets and create **models** by using the below algorithms provided by **sklearn**:    

    - Random Forest --> 72% ACC
    - Multinomial NB --> 35% ACC
    - SGD classsifier --> 59% ACC

For reference **Dummy Classifier** provided a very small accuracy of 6%.    



# PART 3 - Classification with Neural Networks

In this part I use the **tensorflow** library to perform a classification using NNs.   

I use 20% (90000 samples) of the dataset for testing, 12% (54000 samples) for validation and the remaining 68% (306000 samples) for training.   

Before I import the data into the NN I need to convert text(speech) into an array of numbers.   

To do that I:   

    - Define the max length of each speech to 150 words
    - Tokenize each word in order for each word to have a unique code
    - Convert speeches to arrays of integers and use padding at the end

Now the data is ready to be import into the Neural Network.   
To construct my NN I used one Embeding layer followed by one LSTM layer and two Dense.  

After training the model for 10 epochs a multi-classification with accuracy of 80% was achieved.  



