# tutorial code snippets

the following are a collection of code examples from our ML study group.

## necessary packages:
- `numpy`
- `scipy`
- `nltk`
- `sklearn`
- `pandas` (for data manipulation)
- `tensorflow` or `tensorflow-gpu`
- `keras`
- `h5py` (hdf5 for python - lets you save model weights in compact form)
- `gym` (openAI gym)

## extras:
- `gensim` for embeddings
- `jupyter notebook` for easy debugging/sharing
- `mlxtend` (extended machine learning toolkit)


## Tool Tutorials:

### Word2Vec embeddings with gensim: 

read: https://rare-technologies.com/word2vec-tutorial/

see: `embedding.py`

extra: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

### using custom embeddings in keras network: 

read: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

see: `cnn_custom_embeddings.py`

### exploratory data analysis with `pandas`:

read: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/


## 0. basic linear algebra and python classes

watch: https://www.udacity.com/course/linear-algebra-refresher-course--ud953

see: `linalg_functions_blank.py` for template, `linalg_functions.py` for solutions


## 1. basic feed-forward neural network in python

watch: https://www.youtube.com/watch?v=WZDMNM36PsM&t=245s

see: `mycounter.py`for template, fuller network at `simple_keras_counter.py`


## 2. ML classification with `scikit-learn`

read: http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html

see: `simple_classifiers.py`, `brown_corp_generator.py` to generate dataset

also: `cluster.ipynb` jupyter notebook for LSA/LDA

extra: `FeatureUnion` for multiple inputs:

http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

http://michelleful.github.io/code-blog/2015/06/20/pipelines/


## 3. neural classification with `keras`

read: 
http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

see: `simple_rnn_classifier.py`

extra:
http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/

https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/

https://offbit.github.io/how-to-read/

extra:
**Computerphile** videos on the CNN:

https://www.youtube.com/watch?v=py5byOOHZM8

https://www.youtube.com/watch?v=BFdMrDOx_CM


## 4. LSTM language model for text generation

read: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

see: `languagemodel.py`


## 5. sequence-to-sequence `keras` network for simple addition

read: http://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/

see: `simple_seq2seq_demo.py`

source: https://gist.github.com/rouseguy/1122811f2375064d009dac797d59bae9


## 6. reinforcement learning with VSLA

NB: programmed as demo for presentation for reinforcement learning class:
https://bi.snu.ac.kr/Courses/ann16f/presenter/LA_Derek.pptx

read:
https://theses.lib.vt.edu/theses/available/etd-5414132139711101/unrestricted/ch3.pdf

http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable

https://www.researchgate.net/figure/225274789_fig10_Figure-2-Pseudo-code-of-variable-structure-learning-automaton

see: `VSLA_demo.py`


## 7. reinforcement (Q-)learning with `tensorflow`

read: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

see: `simpleRL_0_table.py`, `simpleRL_0_ffnn.py`


## Notes, References and Links:

### tools and stuff:
pandas (for data manipulation)
mlxtend (extended machine learning toolkit)

### word embeddings:
http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

http://ruder.io/word-embeddings-1/

https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

### documentation:
http://scikit-learn.org/stable/index.html

https://keras.io

### preprocessing:
http://www.nltk.org/howto/stem.html

http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

### random forest:
https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

http://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics

### multiclass tfidf:
https://gist.github.com/prinsherbert/92313f15fc814d6eed1e36ab4df1f92d

### gradient boosting:
http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/

http://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

### how to deal with imbalanced classes:
http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/

http://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/

http://stackoverflow.com/questions/15065833/imbalance-in-scikit-learn
