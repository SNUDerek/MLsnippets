import codecs, re, random
from collections import Counter
import numpy as np


# indexes sentences by vocab frequency list
# reserves 0 for UNKs
# todo: probably shoulda used sklearn.####vectorizer

# USAGE
# first get lists like this:
# sents, classes = dataset.get_lists(sents_filename, classes_filename)
# then run train-test split like this:
# train_X, train_y, test_X, test_y, test_set, class_set = \
#     dataset.get_test_train(sents, classes, trainsize=0.8, max_vocab=50000):

# function to get lists from data
# takes corpus as filename (headlines, articles on alternate lines)
# returns lists of sentence token lists, classes
def get_lists(file_corpus, testing=0):
    if testing == 1:
        print('starting dataset.get_lists()...')
    f_corpus = codecs.open(file_corpus, 'rb', encoding='utf8')
    sents = []
    heads = []
    counter = 0

    for line in f_corpus:
        if counter % 2 == 0:
            heads.append(line.strip('\n').split(' '))
        else:
            sents.append(line.strip('\n').split(' '))
        counter += 1
    return(sents, heads)


def get_texts(file_corpus, testing=0):
    if testing == 1:
        print('starting dataset.get_lists()...')
    f_corpus = codecs.open(file_corpus, 'rb', encoding='utf8')
    sents = []
    heads = []
    counter = 0

    for line in f_corpus:
        if counter % 2 == 0:
            heads.append(line.strip('\n'))
        else:
            sents.append(line.strip('\n'))
        counter += 1
    return(sents, heads)

# function to get vocab, maxvocab
# takes list : sents
def get_vocab(sents, heads, testing=0):
    if testing == 1:
        print('starting dataset.get_vocab()...')
    # get vocab list
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word)
    for sent in heads:
        for word in sent:
            vocab.append(word)

    counts = Counter(vocab) # get counts of each word
    vocab_set = list(set(vocab)) # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: -counts[x]) # sort by counts

    if testing==1:
        print("get_vocab[:10]:", sorted_vocab[:10])

    return(sorted_vocab)

# function to convert sents to vectors
# takes list : sents, int : max vocab
# returns list of vectors (as lists)
def vectorize_sents(sents, vocab, max_vocab, testing=0):
    if testing==1:
        print("starting vectorize_sents()...")
    # get sorted vocab
    vectors = []
    # iterate thru sents
    for sent in sents:
        sent_vect = []
        for word in sent:
            idx = vocab.index(word) + 1 # reserve 0 for UNK / OOV
            if idx < max_vocab: # in max_vocab range
                sent_vect.append(idx)
            else: # out of max_vocab range
                sent_vect.append(0)
        vectors.append(sent_vect)
    if testing==1:
        print("vectorize_sents[:10]:", vectors[:10])
    return(vectors)

def onehot_vectorize_sents(sents, vocab, max_vocab, testing=0):
    if testing==1:
        print("starting vectorize_sents()...")
    # get sorted vocab
    vectors = []
    # iterate thru sents
    for sent in sents:
        sent_vect = []
        for word in sent:
            one_hot = []
            idx = vocab.index(word) + 1 # reserve 0 for UNK / OOV
            for i in range(max_vocab+1):
                if i == idx: # matching
                    one_hot.append(1)
                else:
                    one_hot.append(0)
            sent_vect.append(one_hot)
        vectors.append(sent_vect)
    if testing==1:
        print("onehot_vectorize_sents[:10]:", vectors[0])
    return(vectors)

# function to randomize and test-train split
# takes sent list, class list
# returns train sents, train heads, test sents, test heads
def get_test_train(sents, heads, trainsize=0.8, max_vocab=25000, testing=0):

    vocab = get_vocab(sents, heads, testing=testing)
    sent_vectors =  vectorize_sents(sents, vocab, max_vocab, testing=testing)
    head_vectors =  onehot_vectorize_sents(heads, vocab, max_vocab, testing=testing)

    # get list entry ... list?
    entries = []
    for i in range(len(sent_vectors)):
        entries.append(i)

    # shuffle indices for randomization
    shuffled = random.sample(entries, len(entries))
    # stop size for train set
    train_stop = int(len(shuffled)*trainsize)

    train_X = []
    train_y = []
    test_X = []
    test_y = []

    for j in range(len(shuffled)):
        idx = shuffled[j] # get random index
        if j < train_stop:
            train_X.append(sent_vectors[idx])
            train_y.append(head_vectors[idx])
        else:
            test_X.append(sent_vectors[idx])
            test_y.append(head_vectors[idx])

    return(train_X, train_y, test_X, test_y)