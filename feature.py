#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re


# 연속하는 1개의 자질
def index2feature(sent, i, offset):
    index = i + offset

    if offset < 0:
        if index < 0:
            word = ' '
        else:
            word = sent[index][0]

        return '{}:word={}'.format(offset, word)
    else:
        L = len(sent)
        if index >= L:
            word = ' '
        else:
            word = sent[index][0]

        sign = '+'

        return '{}{}:word={}'.format(sign, offset, word)


# 연속하는 2개의 자질
def index2feature2(sent, i, offset):
    word = str()
    # 왼쪽 윈도우
    if offset < 0:
        for index in range(i + offset, i + offset + 2):
            if index < 0:
                word += ' '
            else:
                word += sent[index][0]

        return '{}:word[{}:{}]={}'.format(offset, offset, offset+1, word)

    # 오른쪽 윈도우
    elif offset > 0:
        L = len(sent)
        for index in range(i + offset - 1, i + offset + 1):
            if index >= L:
                word += ' '
            else:
                word += sent[index][0]

        sign = '+'
        return '{}{}:word[{}:{}]={}'.format(sign, offset, offset-1, offset, word)


# 연속하는 3개의 자질
def index2feature3(sent, i, offset):
    word = str()
    # 왼쪽 윈도우
    if offset < 0:
        for index in range(i + offset, i + offset + 3):
            if index < 0:
                word += ' '
            else:
                word += sent[index][0]

        return '{}:word[{}:{}]={}'.format(offset, offset, offset+2, word)

    # 오른쪽 윈도우
    elif offset > 0:
        L = len(sent)
        for index in range(i + offset - 2, i + offset + 1):
            if index >= L:
                word += ' '
            else:
                word += sent[index][0]

        sign = '+'
        return '{}{}:word[{}:{}]={}'.format(sign, offset, offset-2, offset, word)

    # offset == 0
    else:
        L = len(sent)
        for index in range(i + offset - 1, i + offset + 2):
            if index < 0 or index >= L:
                word += ' '
            else:
                word += sent[index][0]

        sign = '+'
        return '{}{}:word[{}:{}]={}'.format(sign, offset, offset-1, offset+1, word)


# Feature 1
def append_feature_1(features, sent, i):
    features = np.append(features, index2feature(sent, i, -2))
    features = np.append(features, index2feature(sent, i, -1))
    features = np.append(features, index2feature(sent, i, 0))
    features = np.append(features, index2feature(sent, i, 1))
    features = np.append(features, index2feature(sent, i, 2))

    return features


# Feature 2
def append_feature_2(features, sent, i):
    features = np.append(features, index2feature2(sent, i, -2))
    features = np.append(features, index2feature2(sent, i, -1))
    features = np.append(features, index2feature2(sent, i, 1))
    features = np.append(features, index2feature2(sent, i, 2))

    return features


# Feature 3
def append_feature_3(features, sent, i):
    features = np.append(features, index2feature3(sent, i, -2))
    features = np.append(features, index2feature3(sent, i, 0))
    features = np.append(features, index2feature3(sent, i, 2))

    return features


def word2features(sent, i):
    features = np.array([], dtype=np.str)
    features = append_feature_1(features, sent, i)
    features = append_feature_2(features, sent, i)
    features = append_feature_3(features, sent, i)

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def text2features(textfile):
    result = []
    with open(textfile, 'r') as rfile:
        for sent in rfile:
            for feature in sent2features(get_no_space_sent(sent)):
                #result.append(np.array([feature]))
                result.append(feature)

    return np.array(result)


def get_no_space_sent(sent):
    return re.sub(r'\s+', '', sent).strip()


def sent2labels(sent):
    # convert input text into tagged form
    text = re.sub(r'\s+', ' ', sent).strip()
    labels = np.array([], dtype=np.int32)
    for i in range(len(text)):
        if i == 0:
            labels = np.append(labels, 2)
        elif text[i] != ' ':
            prevWord = text[i - 1]
            if prevWord == ' ':
                labels = np.append(labels, 2)
            else:
                labels = np.append(labels, 1)

    return labels


def text2labels(textfile):
    result = []
    with open(textfile, 'r') as rfile:
        for sent in rfile:
            for label in sent2labels(sent):
                #result.append(np.array([label]))
                result.append(label)

    return np.array(result, dtype=np.int32)


def text2sent(line):
    # convert input text into tagged form
    text = re.sub(r'\s+', ' ', line).strip()
    sentence = []
    for i in range(len(text)):
        if i == 0:
            sentence.append([text[i], 'B'])
        elif text[i] != ' ':
            prevWord = text[i - 1]
            if prevWord == ' ':
                sentence.append([text[i], 'B'])
            else:
                sentence.append([text[i], 'I'])

    return sentence
