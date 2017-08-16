# this file generates sample classification data from the brown corpus
# it saves the sentence(s) and genres into separate files
# edit the topics list below to select topics

import codecs, re
from nltk.corpus import brown
import pandas as pd

# ################## edit params here ##################

lower_lim = 8       # minimum words per sentence
upper_lim = 100     # maximum words per sentence
max_ex = 1000       # maximum examples per genre
max_clusters = 5    # sentences per example

sents = codecs.open('datasets/brown_sents.txt', 'w', encoding='utf-8')
classes = codecs.open('datasets/brown_topics.txt', 'w', encoding='utf-8')

topics = ['religion', 'government', 'romance', 'news', 'science_fiction']

'''
choose some from the following genres:
adventure
belles_lettres
editorial
fiction
government
hobbies
humor
learned
lore
mystery
news
religion
reviews
romance
science_fiction
'''

striplist = ["`", "'", '!', '?', '.', ',', ':', ';', '-', '(', ')', ]

counts_list = []

csv_sents = []
csv_labels = []

for topic in topics:
    good_count = 0 # for counting good sentences
    this_counter = 0
    this_cluster = ''
    for sentence in brown.sents(categories=[topic]):

        # check length first:
        if lower_lim < len(sentence) < upper_lim:

            this_string = ' '.join(sentence).lower() # lowercase
            for shit in striplist:
                this_string = this_string.replace(shit, '')     # remove punctuation etc
            this_string = re.sub(r'\d', '#', this_string) # sub # for digits
            this_string = re.sub(r'[\s]+', ' ', this_string)

            if this_counter < max_clusters:
                this_cluster += this_string
                this_counter += 1
            else:
                good_count += 1
                sents.write(this_cluster)
                csv_sents.append(this_cluster)
                this_cluster = ''
                sents.write('\n')
                classes.write(topic)
                classes.write('\n')
                csv_labels.append(topic)
                this_counter = 0

            print(good_count, "sentence (clusters) for", topic)
            if good_count > max_ex:
                break

    counts_list.append(good_count)

dicto = {'document' : csv_sents,
         'topic' : csv_labels}
df = pd.DataFrame.from_dict(dicto)
df.to_csv('datasets/brown.csv', sep='\t')

print(sum(counts_list), counts_list)

