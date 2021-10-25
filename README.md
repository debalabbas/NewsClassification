

Interos

ML Apprentice Coding Challenge

Debal Husain Abbas





INTRODUCTION

In this project, I will try to classify the news articles into one of the following classes:

• acq

• crude

• earn

• grain

• interest

• money-fx

• money-supply

• others

• ship

• sugar

• trade





Understanding the Data

The data is in the form of a list of ‘.sgm’ files. Below is an image of how the files

are structured. Each file contains 1000 News Articles.You can find more details

about it in the README file regarding the tags.





Fetching the Data

• I have used the LEWISSPLIT attribute in the REUTERS tag to split the data

into Train and Test Set. If LEWISPLIT = ‘Test’ then the article belongs to

Test set otherwise to Train Set.

• We have extracted Topic(First one if Multiple Exists) from TOPICS tag. If no

topic is assigned then ‘N/A’ is assigned as it’s Topic.

• Extracted the Content of Article from TEXT tag.

• Finally Created two dataframes of Train and Test Set having two attributes

Article and Topic.





View of how the Training

Dataset looks like





Cleaning the Data

We will now check the distribution of Topics in the Train set,

to see which is the most occurring topic in our dataset.

Image on the right shows the top 20 most occurring Topics

in our training set.

As we can see the 7000+ articles have no topic assigned to

them.

Since we are going to use a Supervised Learning Algorithm

we will drop those articles which have no topic assigned to

them.





We have done a multiclass Classification here on **11**

classes

We have selected top **10** highest occurring topics and

assigned topic **others** to the remaining articles.

After doing the transformation this is the distribution of the

topics.

Then we have encoded the Topic variable using [sklearn’s](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

[Label](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)[ ](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)[Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).





View of the Final Dataframe





Then we have created [Datasets](https://huggingface.co/docs/datasets/)[ ](https://huggingface.co/docs/datasets/)object using from\_pandas method for

train and test set.

We are doing as it is faster to train on has various inbuilt functions for

preprocessing it.

Image of the datasets object





PROPOSED METHODOLOGY

We are going to use a Transformer Model to do the classification here.

Transformers are already pre trained on large corpuses of text and require fine

tuning on downstream task to give very good results.

We will attach a Dense layer having 11 neurons and softmax activation on top of it

to get our final model for training and predictions.





Preprocessing

• We will now prepare the article text to be input into the model.

• This will be done by tokenizing and encoding the text . This all is handled by

the huggingface [Tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html)[.](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[It](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[tokenizes](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[the](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[text](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[using](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[the](https://huggingface.co/transformers/main_classes/tokenizer.html)[ ](https://huggingface.co/transformers/main_classes/tokenizer.html)[sentencepiece](https://github.com/google/sentencepiece)

tokenizer and then encodes it and also does padding.

• Padding is done on sentences to make them uniform in length as the deep

learning model will only accept a batch of uniform sequences having same

length.

• It returns two objects **input\_ids** and **attention\_mask**. input\_ids contains the

encoded text while attention mask is of the same length as input ids and the

items in it can be either 0 or 1 , it helps the model to understand whether the

token is part of original text or a padding token.

• Since the model only accepts sequences up to a sequence length of 510 so we

will truncate the sequence to that length, so it can be given as an input to the

model.





**Image of how the final model looks like and make predictions**





Training the Model

Now we will train the model.

• We have used a batch size of **16**(Colab Constraint) and learning rate is set to

be **2\*105**.

• We will train the model for **3** epochs and save the model after each epoch to

load the best one at the end.

• We have used accuracy as the metric to select the best model.

• We have used Categorical Cross Entropy as our loss function.

• We have achieved an accuracy of **94.02%** after 3 epochs on the test set.





Training the Model

Now we will train the model.

• We have used a batch size of **16**(Colab Constraint) and learning rate is set to

be **2\*105**.

• We will train the model for **3** epochs and save the model after each epoch to

load the best one at the end.

• We have used accuracy as the metric to select the best model.

• We have used Categorical Cross Entropy as our loss function.

• We have achieved an accuracy of **94.02%** after 3 epochs on the test set.





Predicting on New Data.

To predict on new data the text of article will be processed using the same

tokenizer, and passed through the model which will return an array of size 11(for

each article). Each element in the array is the confidence score for each class.

We will now return of the index of the element having the highest index score which

will be the prediction.

To decode it we can use the same Label Encoder and use it’s **inverse\_transform**

function, to get it in a more understandable format.

**Note:** I have added a Prediction notebook in the folder to get predictions using the

above trained model and tokenizer.





THANK YOU

