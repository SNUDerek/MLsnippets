import codecs, re, random
from collections import Counter
import numpy as np
from separator import Separator
from konlpy.tag import Mecab

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
def index_sents(sents, vocab, max_vocab, testing=0):
    if testing==1:
        print("starting vectorize_sents()...")
    # get sorted vocab
    vectors = []
    # iterate thru sents
    for sent in sents:
        sent_vect = []
        sentlist = sent.split(' ')
        for word in sentlist:
            if word in vocab.keys():
                idx = vocab[word] + 1 # reserve 0 for UNK / OOV
                if idx < max_vocab: # in max_vocab range
                    sent_vect.append(idx)
            else: # out of max_vocab range or OOV
                sent_vect.append(0)
        vectors.append(sent_vect)
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


# function to return lists of graphemes
# takes sentence as string
# returns list of graphemes
def grapheme_splitter(sent):
    sentlist = sent.strip().split(' ')
    graphlist = []
    for word in sentlist:
        wordlist = list(word)
        for syllable in wordlist:
            # find korean words
            if re.findall(r'[[\uac00-\ud7a3]|[\u1100-\u11ff]]+', syllable):
                graphlist += Separator(syllable).sep_all
            else:
                graphlist.append(syllable)
    return graphlist


# function to return lexicalized morphs from mecab
# takes sentence as string
# returns space-separated string of lexicalized morphemes
def mecab_tokenize(sent):


    return

def kkma_tokenize(sents):
    from konlpy.tag import Kkma
    kkma = Kkma()
    lex_sents = []
    # POS-tag and get lexical form from morphemes using KONLPY
    for sent in sents:
        lex_sents.append(' '.join(kkma.morphs(sent)))
        if len(lex_sents) % 200 == 0:
            print("kkma: done", len(lex_sents), "of", len(sents), "total")
    return lex_sents