import numpy as np
import fasttext
import os
import csv
from gensim.models import Word2Vec,FastText
import gensim.downloader as api
from itertools import chain
from keras.preprocessing.sequence import pad_sequences


class EmbeddingLayer():
    def __init__(self, mode='word2vec'):
        self.type = mode
        self.word2vec_model = Word2Vec(vector_size=300, min_count=1)

        self.word_embeddings = []
        self.X = []
        self.y = []

    def fit_transform(self, data, labels):
        self.data = data
        self.labels = labels
        
        if self.type == 'word2vec':
            self.word2vec()
        elif self.type == 'glove':
            self.glove()
        elif self.type == 'fasttext':
            self.fasttext()
        else:
            raise ValueError('Invalid embed mode')
        
        maxlen = 0
        for i in range(len(self.X)):
            self.X[i] = np.array(self.X[i])
            maxlen = max(maxlen, self.X[i].shape[0])

        self.maxlen = maxlen
        
        for i in range(len(self.X)):
            self.X[i] = np.pad(self.X[i], ((0,maxlen - self.X[i].shape[0]),(0,0)), 'constant')
        
        self.y = self.create_tag_map(self.labels)
        self.y = pad_sequences(self.y,maxlen=maxlen,padding='post',value=0,dtype='int32')

        self.X = np.array(self.X).astype('float32')
        self.y = np.array(self.y).astype('float32')

        return self.X, self.y
    __call__ = fit_transform

    def transform(self, data, labels):
        word_embeddings = []
        
        for i in range(len(data)):
            sentence = data[i]
            embeddings = []
            for word in sentence:
                try:
                    if self.type == 'word2vec':
                        embeddings.append(self.word2vec_model.wv.get_vector(word))
                    elif self.type == 'glove':
                        embeddings.append(self.new_glove_model[word])
                    elif self.type == 'fasttext':
                        embeddings.append(self.fasttext_model[word])
                    else:
                        raise ValueError('Invalid embed mode')
                except KeyError:
                    embeddings.append(np.random.rand(300)*0.1)
            word_embeddings.append(embeddings)
        
        for i in range(len(word_embeddings)):
            word_embeddings[i] = word_embeddings[i][:self.maxlen]
            word_embeddings[i] = np.pad(word_embeddings[i], ((0,self.maxlen - len(word_embeddings[i])),(0,0)), 'constant')
        word_embeddings = np.array(word_embeddings).astype('float32')
        
        if labels is not None:
            y = self.create_tag_map(labels)
            y = pad_sequences(y,maxlen=self.maxlen,padding='post',value=0,dtype='int32')
            y = np.array(y).astype('float32')
        else:
            y = None

        return word_embeddings, y
                        
    def word2vec(self):
        self.word2vec_model.build_vocab(self.data)
        self.word2vec_model.wv.vectors_lockf = np.ones(shape=(len(self.word2vec_model.wv.index_to_key),1))
        self.word2vec_model.wv.intersect_word2vec_format(
            './embeddings/GoogleNews-vectors-negative300.bin',
            binary=True,
            lockf=1.0
        )
        self.word2vec_model.train(self.data, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.epochs, compute_loss=True)

        for sentence in self.data:
            embeddings = []
            for word in sentence:
                embeddings.append(self.word2vec_model.wv.get_vector(word))
            self.word_embeddings.append(embeddings)

        self.X = self.word_embeddings
    
    def glove(self):
        # create a pre-trained word to embedding dictionary
        def glove2dict(glove_filename):
            with open(glove_filename, encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)
                embed = {line[0]: np.array(list(map(float, line[1:])))
                        for line in reader}
            return embed

        glove_path = './embeddings/glove/glove.6B.300d.txt'
        if os.path.exists(glove_path):
           glove_dict = glove2dict(glove_path)
        else: 
             glove_model = api.load("glove-wiki-gigaword-300")
             glove_dict = {word: glove_model[word] for word in glove_model.index_to_key}

        new_glove_embedding_dict=glove_dict
        self.old_glove_model=glove_dict
        self.new_glove_model=new_glove_embedding_dict

        self.word_embeddings = []
        for sentence in self.data:
            embeddings = []
            for word in sentence:
                try:
                    embeddings.append(new_glove_embedding_dict[word])
                except KeyError:
                       embeddings.append(np.random.rand(300)*0.1)
            self.word_embeddings.append(embeddings)
        self.X = self.word_embeddings

    def fasttext(self):
        ft = fasttext.load_model('./embeddings/cc.en.300.bin')
        # ft.train(self.data, total_examples=len(self.data), epochs=10)
        self.fasttext_model = ft

        for sentence in self.data:
            embeddings = []
            for word in sentence:
                embeddings.append(self.fasttext_model[word])
            self.word_embeddings.append(embeddings)
        self.X = self.word_embeddings

    def create_tag_map(self,labels):
        y_un = list(chain(*labels))
        y_un = list(set(y_un))
        tag_map = {}
        for i in range(len(y_un)):
            tag_map[y_un[i]] = (i+1)
        
        for label in labels:
            for i in range(len(label)):
                label[i] = tag_map[label[i]]
        return labels