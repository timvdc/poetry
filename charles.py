#!/usr/bin/env python

import sys
from subprocess import Popen, PIPE
import random
from countsyl import count_syllables
import time
import numpy as np
import pickle as pickle
from threading  import Thread
from queue import Queue, Empty
import os
#from keras.models import model_from_json
#import countsyl
import numpy as np
import scipy.stats
import kenlm
#import theano
from datetime import datetime
import codecs
import warnings
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from functools import reduce
import copy
from poemutils import hmean
from pprint import pprint
import onmt
import argparse
import torch

#this one for RTX (cudnn too old)
torch.backends.cudnn.enabled = False

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))


#from nmt import (load_params, init_params, init_tparams, build_model, prepare_data)

#warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.filterwarnings("ignore")

VOCAB_FILE_KERAS = '/data/cruys/work/neural/rnn_lstm/frcow/keras/vocab_keras.15000'
VOCAB_FILE_SEQ = '/data/cruys/work/neural/rnn_lstm/frscrape/utf8/preproc/vocab.15000'
#NMF_FILE = '/data/cruys/work/poetry/fr/nmf/utf8/p_wd2_d100.npy'
NMF_FILE = 'data/p_wd2_d100.npy'
NMF_DESCRIPTION_FILE = '/data/cruys/work/poetry/fr/v5/nmf/all/description_nmf_d100_it100.pickle'

RHYME_FREQ_FILE = 'data/rijm_fr_small.freq'
RHYME_DICT_FILE = 'data/rhymeDictionary_fr.pickle'
RHYME_INV_DICT_FILE = 'data/rhymeInv_fr_utf8.pickle'

#MODEL_FILE = '/data/cruys/work/neural/rnn_lstm/frcow/keras/rnnlm_rev_2gru.json'
#PARAM_FILE = '/data/cruys/work/neural/rnn_lstm/frcow/keras/rnnlm_2gru1024_w15k_rev_emb.h5_finetune2'
MODEL_FILE = '/data/cruys/work/neural/pytorch/OpenNMT-py_adapt/own/fr_aran_full_rev-model_e512_d2048_general_acc_0.00_ppl_28.61_e25.pt'

class Poem:

    def __init__(self, form='short', init=False):

        self.structureDict = {'sonnet':
                              ('a','b','b','a', '',
                               'c','d','d','c', '',
                               'e','f','e', '',
                               'f', 'e', 'f'),
                              'short':
                              ('a','b','a','b', '',
                               'c','d','c','d'),
                              'shorter':
                              ('a','b','a','b'),
                              }
        self.form = form
        self.loadRhymeDictionary()
        self.loadNMFData()

        #self.generator = ReverseLM(MODEL_FILE, PARAM_FILE, self.i2w, self.w2i)
        self.generator = VerseGenerator(MODEL_FILE)
    
        self.loadVocabulary()

        #self.ngramModel = kenlm.Model('../ngram/frscrape_3gram.binary')
        self.ngramModel = kenlm.Model('/data/cruys/work/poetry/fr/v5/ngram/corpus_all_3gram.binary')
        
        #self.encdecScorer = EncDecScorer()

        logfile = 'log/poem_' + datetime.now().strftime("%Y%m%d")
        self.log = open(logfile, 'a')


    def loadNMFData(self):
        self.W = np.load(NMF_FILE)
        with open(NMF_DESCRIPTION_FILE, 'rb') as f:
            self.nmf_descriptions = pickle.load(f, encoding='utf8')
        
    def loadRhymeDictionary(self):
        freqRhyme = {}
        with codecs.open(RHYME_FREQ_FILE, 'r', encoding='utf8') as f:
            for line in f:
                line = line.rstrip()
                rhyme, freq = line.split('\t')
                freqRhyme[rhyme] = int(freq)
        self.freqRhyme = freqRhyme
        self.rhymeDictionary = pickle.load(open(RHYME_DICT_FILE, 'rb'))
        self.rhymeInvDictionary = pickle.load(open(RHYME_INV_DICT_FILE, 'rb'))

    def loadVocabulary(self):
        i2w,w2i = [], {}
        #i2wseq, w2iseq = [], {}

        #with open(VOCAB_FILE_KERAS) as f:
        #    for line in f:
        #        line = line.rstrip()
        #        w2i[line] = len(i2w)
        #        i2w.append(line)
        #self.i2w = i2w
        #self.w2i = w2i
        self.i2w = self.generator.vocab.itos
        self.w2i = self.generator.vocab.stoi

        # with open(VOCAB_FILE_SEQ) as f:
        #     for line in f:
        #         line = line.rstrip()
        #         w2iseq[line] = len(i2wseq)
        #         i2wseq.append(line)
        # self.i2wseq = i2wseq
        # self.w2iseq = w2iseq

    def write(self, constraints=('rhyme'), nmfDim=False):
        self.blacklist_words = set()
        self.blacklist = []
        self.previous_sent = None
        if constraints == ('rhyme'):
            self.writeRhyme(nmfDim)

    def writeRhyme(self, nmfDim):
        rhymeStructure = self.getRhymeStructure()
        #blacklist = []
        #self.log.write(datetime.now().strftime("%Y%m%d-%H%M%S") + '\n\n')
        if nmfDim == 'random':
            nmfDim = random.randint(0,self.W.shape[1] - 1)
        elif type(nmfDim) == int:
            nmfDim = nmfDim
        else:
            nmfDim = None
        if nmfDim:
            sys.stdout.write('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +' nmfdim ' + str(nmfDim) + ' (' + ', '.join(self.nmf_descriptions[nmfDim]) + ')\n\n')
            self.log.write('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' nmfdim ' + str(nmfDim) + ' (' + ', '.join(self.nmf_descriptions[nmfDim]) + ')\n\n')
        else:
            sys.stdout.write('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' NO nmfdim' + '\n\n')
            self.log.write('\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' NO nmfdim' + '\n\n')
        for el in rhymeStructure:
            if el:
                try:
                    words = self.getSentence(rhyme=el, syllables = True, nmf=nmfDim)
                except KeyError as e:
                    print('err', e)
                    continue
                else:
                    sys.stdout.write(' '.join(words) + '\n')
                    #self.typeString(' '.join(words) + '\n')
                    self.log.write(' '.join(words) + '\n')
                    try:
                        self.blacklist.append(self.rhymeDictionary[words[-1]])
                        self.blacklist_words = self.blacklist_words.union(words)
                    except KeyError as e:
                        #means verse does not follow rhyme, probably because of entropy computations
                        #do not show error for presentation
                        #print('err blacklist', e)
                        pass
                    except IndexError as e2:
                        print('err blacklist index', e2)
                    self.previous_sent = words
            else:
                sys.stdout.write('\n')
                self.log.write('\n')
        self.signature()
        self.log.write('\n\n')
        self.log.flush()

    def getSentence(self, rhyme, syllables, nmf):
        if self.previous_sent:
            previous = self.previous_sent
        else:
            previous = None
        if rhyme:
            rhymePrior = self.createRhymeProbVector(rhyme)
        else:
            rhymePrior = None
        if nmf:
            nmfPrior = copy.deepcopy(self.W[:,nmf])
        else:
            nmfPrior = None
        #prior2[0:99] = max(prior2) * .05
        allCandidates = []
        allProbScores = []
        allEncDecScores = []
        for batch, probScores in self.generator.generate_candidates(previous=previous,rhymePrior=rhymePrior, nmfPrior=nmfPrior):
        #for batch, probScores in self.generator.generate_candidates():
            allCandidates.extend(batch)
            allProbScores.extend(probScores)
            #if self.previous_sent:
            #    edScores = self.encdecScorer.score(batch, self.previous_sent)
            #    allEncDecScores.extend(edScores)
        #print allEncDecScores
        #if self.previous_sent:
        #    allEncDecScores = - np.array(allEncDecScores)
        #    largest = allEncDecScores[np.argmax(allEncDecScores)]
        #    allEncDecNorm = np.exp(allEncDecScores - largest)

            #zz = zip(allEncDecNorm, allCandidates)
            #zz.sort()
            #print self.previous_sent, zz

        ngramScores = []
        for ncand, candidate in enumerate(allCandidates):
            try:
                ngramScore = self.ngramModel.score(' '.join(candidate)) / len(candidate)
            except ZeroDivisionError:
                ngramScore = -100
            #print candidate, ngramScore
            ngramScores.append(ngramScore)
        ngramScores = np.array(ngramScores)
        largest = ngramScores[np.argmax(ngramScores)]
        ngramNorm = np.exp(ngramScores - largest)

        #encdecScores = np.array(allProbScores)
        #largest = encdecScores[np.argmax(encdecScores)]
        #encdecNorm = np.exp(encdecScores - largest)

        #for ncand, candidate in enumerate(allCandidates):
        #    print candidate, ngramNorm[ncand]
        scoreList = []
        for ncand, candidate in enumerate(allCandidates):
            #if self.previous_sent:
                #allScores = [allProbScores[ncand], allEncDecNorm[ncand], ngramNorm[ncand]]
                #allScores = [allProbScores[ncand], ngramNorm[ncand]]
            #else:
            allScores = [allProbScores[ncand], ngramNorm[ncand]]
            #allScores = [encdecNorm[ncand], ngramNorm[ncand]]
            #print allScores
            if syllables:
                syllablesScore = self.checkSyllablesScore(candidate, mean=12, std=2)
                allScores.append(syllablesScore)
            # if rhyme:
            #     rhymeScore = self.checkRhymeScore(words, rhymeSound)
            #     allScores.append(rhymeScore)
            # if ending:
            #     endScore = self.checkEndScore(words)
            #     allScores.append(endScore)
            if nmf:
                NMFScore = self.checkNMF(candidate, [nmf])
                allScores.append(NMFScore)
            # if subject:
            #     subjectScore = self.checkNMF(words,self.subjectDims)
            #     allScores.append(subjectScore)
            
            #allScore = reduce(lambda x,y: float(x)*float(y), allScores)
            #print(allScores)
            allScore = hmean(allScores)
            #allScore = allScores[0]
            scoreList.append((allScore, candidate, allScores))

        scoreList.sort()
        scoreList.reverse()
        #print(scoreList[0])
        #pprint(scoreList[0:10])
        #print('\n'.join('\t'.join(str(i)) for i in scoreList))
        #for i in range(6):
        #    print(i, scoreList[i][1], scoreList[i][0], scoreList[i][2], '\n')
        return scoreList[0][1]
        #length10 = [s for s in allCandidates if len(s) == 10]
        #return random.choice(length10)

    def getRhymeStructure(self, cutoff=5):
        chosenList = []
        mapDict = {}
        structure = self.structureDict[self.form]
        for el in set(structure):
            freq = -1
            while True:
                rhymeForm = random.choice(list(self.freqRhyme.keys()))
                freq = self.freqRhyme[rhymeForm]
                if (freq >= cutoff) and not rhymeForm in chosenList:
                    chosenList.append(rhymeForm)
                    mapDict[el] = rhymeForm
                    break
        rhymeStructure = []
        for struct in structure:
            if struct:
                rhymeStructure.append(mapDict[struct])
            else:
                rhymeStructure.append(struct)
        return rhymeStructure

    def createRhymeProbVector(self, rhyme):
        probVector = np.empty(len(self.i2w))
        probVector.fill(1e-20)
        for w in self.rhymeInvDictionary[rhyme]:
            #print w
            if not self.rhymeDictionary[w] in self.blacklist:
                probVector[self.w2i[w]] = 1
        return probVector / np.sum(probVector)

    def signature(self):
        sys.stdout.write('\n                                     ')
        time.sleep(4)
        for el in '- Charles':
            nap = random.uniform(0.1,0.6)
            sys.stdout.write(el)
            sys.stdout.flush()
            time.sleep(nap)
        sys.stdout.write('\n')

    def typeString(self, verse):
        for el in verse:
            nap = random.uniform(0.1,0.3)
            sys.stdout.write(el)
            sys.stdout.flush()
            time.sleep(nap)
        return None


    def checkSyllablesScore(self, words, mean, std):
        gaussian = scipy.stats.norm(mean,std)
        nSyllables = sum([count_syllables(w)[1] for w in words])
        return gaussian.pdf(nSyllables) / 0.19

    def computeNMFScore(self,words,dimList):
        sm = 0
        sm = sum([max(self.W[self.w2i[w],dimList]) for w in words if w in self.w2i])
        return sm

    def checkNMF(self, words, dimList):
        words = list(set([w for w in words if not w in self.blacklist_words]))
        NMFTop = np.max(np.max(self.W[:,dimList], axis=0))
        NMFScore = self.computeNMFScore(words, dimList)
        return NMFScore / NMFTop
        

class VerseGenerator:
    def __init__(self, modelFile):
        opt = argparse.Namespace(model=modelFile, data_type='text', gpu=0)
        self.fields, self.model, self.model_opt = \
            onmt.ModelConstructor.load_test_model(opt, {})
        self.vocab = self.fields["tgt"].vocab

        self.batch_size_decoder = 20
        self.max_length = 100
        
    def generate_candidates(self, previous, rhymePrior, nmfPrior):
        # process priors once, not in generation loop
        if rhymePrior is not None and nmfPrior is not None:
            # combine both priors
            rhymePrior = rhymePrior * nmfPrior
            rhymePrior = rhymePrior / np.sum(rhymePrior)
        if rhymePrior is not None:
            rhymePrior = torch.tensor(rhymePrior).float()
            rhymePrior = rhymePrior.repeat(self.batch_size_decoder, 1)
        if nmfPrior is not None:
            nmfPrior = torch.tensor(nmfPrior).float()
            nmfPrior = nmfPrior.repeat(self.batch_size_decoder, 1)

        for nbatch in range(50):
            #print('nn', nbatch)

            if previous is None:
                # when no previous verse is defined (first verse of
                # the poem), encode the phrase "unk unk unk" - this
                # works better for initialization of the decoder than
                # an all-zero or random hidden encoder state
                src = torch.tensor([0, 0, 0])

            else:
                src = torch.tensor([self.vocab.stoi[w] for w in previous])
            # we need a three-way tensor, double unsqueeze
            src = src.unsqueeze(1).unsqueeze(2).cuda()
            src_lengths = torch.tensor([src.size(0)]).cuda()


            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)


            #memory_bank = rvar(memory_bank.data)
            memory_bank = memory_bank.repeat(1, self.batch_size_decoder, 1).cuda()
            memory_lengths = src_lengths.repeat(self.batch_size_decoder).cuda()
            dec_states.repeat_beam_size_times(self.batch_size_decoder)

            sampler = onmt.translate2.Sampler(self.batch_size_decoder,
                                              pad=self.vocab.stoi[onmt.io.PAD_WORD],
                                              eos=self.vocab.stoi[onmt.io.EOS_WORD],
                                              bos=self.vocab.stoi[onmt.io.BOS_WORD],
                                              cuda=True
                                           )



            for i in range(self.max_length):
                #print('step', i)
                #inp = var(sampler.get_current_state().t().contiguous()) #.view(1, -1))                                                                                  
                inp = torch.stack([sampler.get_current_state()])
                inp = inp.contiguous().t().view(1,-1)

                if sampler.done():
                    break

                inp = inp.unsqueeze(2).cuda()

                dec_out, dec_states, attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths)
                dec_out = dec_out.squeeze(0)
                #print('d', dec_out.size())

                out = self.model.generator.forward(dec_out).data
                #print('o1', out.size())

                if i == 0 and rhymePrior is not None:
                    sampler.advance(out, prior=rhymePrior.cuda())
                elif nmfPrior is not None:
                    sampler.advance(out, prior=nmfPrior.cuda())
                else:
                    sampler.advance(out)

            # assemble verse candidates and scores
            batch_matrix = torch.stack(sampler.next_ys).cpu().numpy().T
            allCandidates = []
            for i in range(self.batch_size_decoder):
                currSentVector = batch_matrix[i]
                currSentVector = currSentVector[1:np.where(currSentVector == 3)[0][0]]
                currSentWords = [self.vocab.itos[j] for j in currSentVector]
                currSentWords.reverse()
                allCandidates.append(currSentWords)
            #print(allCandidates)
            allScores = list(sampler.scores)
            #print(allScores)
            yield allCandidates, allScores


# class ReverseLM:
#     def __init__(self, modelFile, paramFile, i2w, w2i):
#         with open(modelFile) as f:
#             json_string = f.read()
#         self.model = model_from_json(json_string)
#         self.model.load_weights(paramFile)
#         self.i2w = i2w
#         self.w2i = w2i
        
#     def sample_matrix(self, a, temperature, unk):
#         samples = np.zeros((a.shape[0]), dtype='i')
#         if unk:
#             a[:,unk] = 1e-20
#         a = np.asarray(a).astype('float64')
#         a = np.log(a) / temperature
#         a = np.exp(a) / np.sum(np.exp(a), axis=1)[:,np.newaxis]
#         for i in range(a.shape[0]):
#             samples[i] = np.argmax(np.random.multinomial(1,a[i],1))
#         return samples #, a[list(range(len(samples))),samples]

#     def sample_matrix_prior(self, a, prior, temperature, unk):
#         samples = np.zeros((a.shape[0]),dtype='i')
#         if unk:
#             a[:,unk] = 1e-20
#         a = np.asarray(a).astype('float64')
#         a = a * prior
#         a = a / np.sum(a, axis=1)[:,np.newaxis]
#         a = np.log(a) / temperature
#         a = np.exp(a) / np.sum(np.exp(a), axis=1)[:,np.newaxis]
#         for i in range(a.shape[0]):
#             samples[i] = np.argmax(np.random.multinomial(1,a[i],1))
#         #print samples
#         return samples #, a[range(len(samples)),samples]
        
#     def generate_batch(self, nbatch=128, maxlen=20, temperature = .5, unk=2):
#         #keep list of scores
#         allScores = [[] for _ in range(nbatch)]
#         #number of timesteps
#         sentence_ind = np.zeros(maxlen, dtype=np.int)
#         #prepare batch input
#         sentence_ind = np.tile(sentence_ind, (nbatch,1))

#         sentList = [[] for i in range(nbatch)]
#         for step in range(maxlen):
#             batch_input = sentence_ind
#             preds = self.model.predict(batch_input, verbose=0)
#             next_index = self.sample_matrix(preds, temperature, unk)
#             for i, ni in enumerate(next_index):
#                 allScores[i].append(preds[i,ni])
#             sentence_ind = np.roll(sentence_ind, -1, axis=1)
#             sentence_ind[:,-1] = next_index
#             next_word = [self.i2w[int(i)] for i in next_index]
#             [sentList[i].append(next_word[i]) for i in range(len(sentList))]
#         return sentList, allScores

#     def generate_batch_rhyme_prior(self, prior, nbatch=128, maxlen=20, temperature = .8, unk=2):
#         #keep list of scores
#         allScores = [[] for _ in range(nbatch)]
#         #sentList = []
#         #sentence_ind = [self.w2i[word] for word in sentList]
#         #padding up to number of timesteps
#         #sentence_ind = np.concatenate([np.zeros(maxlen - len(sentList), dtype=np.int8), sentence_ind])
#         #prepare batch input
#         sentence_ind = np.zeros(maxlen, dtype=np.int)
#         sentence_ind = np.tile(sentence_ind, (nbatch,1))
#         sentList = [[] for i in range(nbatch)]
#         #first step with prior rhyme probability
#         batch_input = sentence_ind
#         preds = self.model.predict(batch_input, verbose=0)
#         next_index = self.sample_matrix_prior(preds, prior, temperature=temperature, unk=unk)
#         for i, ni in enumerate(next_index):
#             allScores[i].append(preds[i,ni])
#         sentence_ind = np.roll(sentence_ind, -1, axis=1)
#         sentence_ind[:,-1] = next_index
#         next_word = [self.i2w[int(i)] for i in next_index]
#         #print next_word
#         #print n_probs
#         [sentList[i].append(next_word[i]) for i in range(len(sentList))]
#         #sentProbs = np.log(n_probs)
#         #steps 2-maxlen
#         for step in range(maxlen - 1):
#             batch_input = sentence_ind
#             preds = self.model.predict(batch_input, verbose=0)
#             #next_index, n_probs = self.sample_matrix(preds, temperature, unk)
#             next_index = self.sample_matrix(preds, temperature=temperature, unk=unk)
#             for i, ni in enumerate(next_index):
#                 allScores[i].append(preds[i,ni])
#             sentence_ind = np.roll(sentence_ind, -1, axis=1)
#             sentence_ind[:,-1] = next_index
#             next_word = [self.i2w[int(i)] for i in next_index]
#             [sentList[i].append(next_word[i]) for i in range(len(sentList))]
#             #sentProbs = sentProbs + np.log(n_probs)
#         #largest = sentProbs[np.argmax(sentProbs)]
#         #print largest
#         #print sentList, sentProbs
#         #print sentList, np.exp(sentProbs - largest)
#         #dividing by largest probability in order to normalize, is subtraction of largest probability in log space
#         #sentProbsNorm = np.exp(sentProbs - largest)
#         #s = np.argsort(sentProbs)
#         #for i in s:
#         #    print sentList[i], sentProbsNorm[i]
#         return sentList, allScores#, sentProbs

#     def generate_batch_prior(self, prior, prior2, nbatch=128, maxlen=20, temperature = .8, unk=2):
#         #keep list of scores
#         allScores = [[] for _ in range(nbatch)]
#         #sentList = []
#         #sentence_ind = [self.w2i[word] for word in sentList]
#         #padding up to number of timesteps
#         #sentence_ind = np.concatenate([np.zeros(maxlen - len(sentList), dtype=np.int8), sentence_ind])
#         #prepare batch input
#         sentence_ind = np.zeros(maxlen, dtype=np.int)
#         sentence_ind = np.tile(sentence_ind, (nbatch,1))
#         sentList = [[] for i in range(nbatch)]
#         #first step with prior rhyme probability
#         batch_input = sentence_ind
#         preds = self.model.predict(batch_input, verbose=0)
#         next_index = self.sample_matrix_prior(preds, prior * prior2, temperature=temperature, unk=unk)
#         for i, ni in enumerate(next_index):
#             allScores[i].append(preds[i,ni])
#         sentence_ind = np.roll(sentence_ind, -1, axis=1)
#         sentence_ind[:,-1] = next_index
#         next_word = [self.i2w[int(i)] for i in next_index]
#         #print next_word
#         #print n_probs
#         [sentList[i].append(next_word[i]) for i in range(len(sentList))]
#         #sentProbs = np.log(n_probs)
#         #steps 2-maxlen
#         for step in range(maxlen - 1):
#             batch_input = sentence_ind
#             preds = self.model.predict(batch_input, verbose=0)
#             #next_index, n_probs = self.sample_matrix(preds, temperature, unk)
#             next_index = self.sample_matrix_prior(preds, prior2, temperature=temperature, unk=unk)
#             for i, ni in enumerate(next_index):
#                 allScores[i].append(preds[i,ni])
#             sentence_ind = np.roll(sentence_ind, -1, axis=1)
#             sentence_ind[:,-1] = next_index
#             next_word = [self.i2w[int(i)] for i in next_index]
#             [sentList[i].append(next_word[i]) for i in range(len(sentList))]
#             #sentProbs = sentProbs + np.log(n_probs)
#         #largest = sentProbs[np.argmax(sentProbs)]
#         #print largest
#         #print sentList, sentProbs
#         #print sentList, np.exp(sentProbs - largest)
#         #dividing by largest probability in order to normalize, is subtraction of largest probability in log space
#         #sentProbsNorm = np.exp(sentProbs - largest)
#         #s = np.argsort(sentProbs)
#         #for i in s:
#         #    print sentList[i], sentProbsNorm[i]
#         return sentList, allScores#, sentProbs


#     def generate_candidates(self, prior=None, prior2=None):
#         for i in range(2):
#             #if prior == None:
#             if type(prior) == type(None):
#                 sampledList, allScores = self.generate_batch()
#             #elif prior2 == None:
#             elif type(prior2) == type(None):
#                 sampledList, allScores = self.generate_batch_rhyme_prior(prior)
#             else:
#                 sampledList, allScores = self.generate_batch_prior(prior, prior2)

#             allSent = [i[0:i.index('<EOS>')] if '<EOS>' in i else [] for i in sampledList]
#             [i.reverse() for i in allSent]
#             allSentScores = [(i[0:i.index('<EOS>')], j[0:i.index('<EOS>')]) if '<EOS>' in i else ([], []) for i,j in zip(sampledList, allScores)]
#             #allSentScores = [[np.sum(np.log(j)),i] for i,j in allSentScores]
#             allSentScores = [np.sum(np.log(j)) for i,j in allSentScores]
#             for ns, s in enumerate(allSent):
#                 if not s:
#                     allSentScores[ns] = -10000
#                     #sentProbsNorm[ns] = 0
#             largest = allSentScores[np.argmax(allSentScores)]
#             sentProbsNorm = np.exp(allSentScores - largest)
#             yield allSent, sentProbsNorm

# class EncDecScorer:
#     def __init__(self):
#         #self.model = '/data/cruys/work/neural/rnn_lstm/frscrape2/models/drop5/model_poetry'
#         self.model = '/data/cruys/work/neural/rnn_lstm/frscrape/utf8/models/drop5/model_poetry'
#         with open('%s.pkl' % self.model, 'rb') as f:
#             self.options = pickle.load(f)
        
#         # allocate model parameters
#         self.params = init_params(self.options)
#         # load model parameters and set theano shared variables
#         self.params = load_params(self.model, self.params)
#         self.tparams = init_tparams(self.params)

#         #self.dictionary= '/data/cruys/work/neural/rnn_lstm/frcow/preproc/vocab_15k.pkl'
#         self.dictionary= '/data/cruys/work/neural/rnn_lstm/frscrape/utf8/preproc/vocab_15k_p3.pkl'
#         with open(self.dictionary, 'rb') as f:
#             #self.word_dict = pickle.load(f)
#             self.word_dict = pickle.load(f, encoding='utf8')
#         self.word_idict = dict()
#         for kk, vv in self.word_dict.items():
#             self.word_idict[vv] = kk
#         self.word_idict[0] = '<eos>'
#         self.word_idict[1] = 'UNK'

#         self.f_scorer = self.build_scorer()

#     def build_scorer(self):
#         trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost = build_model(self.tparams, self.options)
#         inps = [x, x_mask, y, y_mask]
#         f_log_probs = theano.function(inps, cost)
#         return f_log_probs

#     def score(self, batch, previous_sent):
#         xxx = np.array([self.word_dict[el] if el in self.word_dict else 1 for el in previous_sent])
#         xx = np.tile(xxx, (len(batch), 1))
#         yy = np.array([[self.word_dict[el] if el in self.word_dict else 1 for el in sent][::-1] for sent in batch])

#         x, x_mask, y, y_mask = prepare_data(xx, yy, maxlen=20, n_words_src=10000,n_words=10000)
#         scores = self.f_scorer(x, x_mask, y, y_mask)
#         lengths = np.array([len(el) for el in yy])
#         scores = scores / lengths
#         return scores
