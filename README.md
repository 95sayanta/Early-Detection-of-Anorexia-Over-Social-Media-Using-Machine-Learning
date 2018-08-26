# Early Detection of Anorexia Over Social Media Using Machine Learning.
You can found the paper from the following link : 
http://ceur-ws.org/Vol-2125/paper_182.pdf

First and foremost, the data that we have dealt with was in XML file. So data preprossing was a significant task towards this early prediction of disorder. We had to extract only the text part from the each of the xml files. Initially the training data consists of two classes, i.e., Anorexia(positive) and non-anorexia(negative). In the positive category, there are 200 xml files(given in 10 chunk where each of the chunks consists of 20 xml files) and in the negative category, there are 1320 xml files(132 xml files in each of the 10 chunks). Here each xml file corresponds to a particular user of a social media user.

Therefore the task consists of identifying whether the posts of a particular person in the test set belong to the anorexia category.

Various text classifiers have been used to accomplish this task based on the said corpus. As you can think of, there are many negative examples as compared to the positive examples. The classifiers that have been used for achieving the goal are:  Support Vector Machine, Random Forest, Logistic Regression, Ada-Boost and Recurrent Neural Network(RNN) as well. The performance of each of the aforementioned classifiers have been have been tested using only the text feature, i.e., Bag-of-Words(BOW), the UMLS(Unified Medical Language System) features and then combination of both BOW and UMLS features.

For implementing UMLS features, we have use MetaMap(https://metamap.nlm.nih.gov/), a tool to recognize UMLS concepts in free-text. Here we have considered only the categories which can be closely related to anorexia. 

Besides, to implement RNN with proper word context, we have uesd GloVe(https://nlp.stanford.edu/pubs/glove.pdf) word embedding.

Hence, if you have any issues regarding this project, e.g., code or concept, please feel free to drop an email at:  sayanta95@gmail.com

Thanks and regards,

Sayanta Paul

Ramakrishna Mission Vivekananda Educational and Research Institute
