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

Word2Vec embeddings with gensim: `embedding.py`

using custom embeddings in keras network: `cnn_custom_embeddings.py`


## 0. basic linear algebra and python classes

watch: https://www.udacity.com/course/linear-algebra-refresher-course--ud953

see: `linalg_functions_blank.py` for template, `linalg_functions.py` for solutions

also: `clustering.ipynb` for unsupervised clustering and `gensim` LDA/LSA


## 1. basic feed-forward neural network in python

watch: https://www.youtube.com/watch?v=WZDMNM36PsM&t=245s

see: `mycounter.py`for template, fuller network at `simple_keras_counter.py`


## 2. ML classification with `scikit-learn`

read:http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html

see: `simple_classifiers.py`, `brown_corp_generator.py` to generate dataset

also: `cluster.ipynb` jupyter notebook for LSA/LDA


## 3. neural classification with `keras`

read: 
http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

see: `simple_rnn_classifier.py`

also:
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

### other stuff:
pandas (for data manipulation)
mlxtend (extended machine learning toolkit)

### documentation:
http://scikit-learn.org/stable/index.html
https://keras.io

### proprocessing:
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
