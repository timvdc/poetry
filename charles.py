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
import numpy as np
import scipy.stats
import kenlm
from datetime import datetime
import codecs
import warnings
from functools import reduce
import copy
from poemutils import hmean
from pprint import pprint
import onmt
import argparse
import torch

#this one for RTX (cudnn too old)
#torch.backends.cudnn.enabled = False

warnings.filterwarnings("ignore")

NMF_FILE = 'data/p_wd2_d100.npy'
NMF_DESCRIPTION_FILE = 'data/description_nmf_d100_it100.pickle'

RHYME_FREQ_FILE = 'data/rijm_fr_small.freq'
RHYME_DICT_FILE = 'data/rhymeDictionary_fr.pickle'
RHYME_INV_DICT_FILE = 'data/rhymeInv_fr_utf8.pickle'

MODEL_FILE = 'data/fr_full_rev-model_e512_d2048_general_acc_0.00_ppl_28.61_e25.pt'

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

        self.generator = VerseGenerator(MODEL_FILE)
    
        self.loadVocabulary()

        self.ngramModel = kenlm.Model('data/corpus_pruned_3gram.binary')

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
        self.i2w = self.generator.vocab.itos
        self.w2i = self.generator.vocab.stoi

    def write(self, constraints=('rhyme'), nmfDim=False):
        self.blacklist_words = set()
        self.blacklist = []
        self.previous_sent = None
        if constraints == ('rhyme'):
            self.writeRhyme(nmfDim)

    def writeRhyme(self, nmfDim):
        rhymeStructure = self.getRhymeStructure()
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
                        #TODO
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
        allCandidates = []
        allProbScores = []
        allEncDecScores = []
        for batch, probScores in self.generator.generate_candidates(previous=previous,rhymePrior=rhymePrior, nmfPrior=nmfPrior):
            allCandidates.extend(batch)
            allProbScores.extend(probScores)

        ngramScores = []
        for ncand, candidate in enumerate(allCandidates):
            try:
                ngramScore = self.ngramModel.score(' '.join(candidate)) / len(candidate)
            except ZeroDivisionError:
                ngramScore = -100
            ngramScores.append(ngramScore)
        ngramScores = np.array(ngramScores)
        largest = ngramScores[np.argmax(ngramScores)]
        ngramNorm = np.exp(ngramScores - largest)

        scoreList = []
        for ncand, candidate in enumerate(allCandidates):
            allScores = [allProbScores[ncand], ngramNorm[ncand]]
            if syllables:
                syllablesScore = self.checkSyllablesScore(candidate, mean=12, std=2)
                allScores.append(syllablesScore)
            if nmf:
                NMFScore = self.checkNMF(candidate, [nmf])
                allScores.append(NMFScore)
            allScore = hmean(allScores)
            scoreList.append((allScore, candidate, allScores))

        scoreList.sort()
        scoreList.reverse()
        return scoreList[0][1]

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


            memory_bank = memory_bank.repeat(1, self.batch_size_decoder, 1).cuda()
            memory_lengths = src_lengths.repeat(self.batch_size_decoder).cuda()
            dec_states.repeat_beam_size_times(self.batch_size_decoder)

            sampler = onmt.translate.Sampler(self.batch_size_decoder,
                                              pad=self.vocab.stoi[onmt.io.PAD_WORD],
                                              eos=self.vocab.stoi[onmt.io.EOS_WORD],
                                              bos=self.vocab.stoi[onmt.io.BOS_WORD],
                                              cuda=True
                                           )



            for i in range(self.max_length):

                inp = torch.stack([sampler.get_current_state()])
                inp = inp.contiguous().t().view(1,-1)

                if sampler.done():
                    break

                inp = inp.unsqueeze(2).cuda()

                dec_out, dec_states, attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths)
                dec_out = dec_out.squeeze(0)

                out = self.model.generator.forward(dec_out).data

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
            allScores = list(sampler.scores)
            yield allCandidates, allScores

