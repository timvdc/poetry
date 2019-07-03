from __future__ import division
import torch
#from onmt.translate2 import Penalties
import numpy as np

class Sampler(object):

    def __init__(self, batch_size, pad, eos, bos, cuda=False):
        self.s_func = torch.nn.Softmax()
        self.batch_size = batch_size
        self.tt = torch.cuda if cuda else torch

        #self.scores = self.tt.FloatTensor(batch_size).zero_()
        self.scores = np.zeros(batch_size)
        self.all_scores = []

        self.next_ys = [self.tt.LongTensor(batch_size)
                        .fill_(bos)]

        # Has EOS topped the beam yet.                                                                                                       
        self._eos = eos
        self.eos_top = False

        self.min_length = 5
        
        # The attentions (matrix) for each time.                                                                                             
        self.attn = []
        
        self.finished = set()
    

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]


    def advance(self, word_probs, prior=None):
        num_words = word_probs.size(1)
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # edit: no unks, zero index hardcoded
        for k in range(len(word_probs)):
            word_probs[k][0] = -1e20
        # Sum the previous scores.
        #if len(self.prev_ks) > 0:
        #    beam_scores = word_probs + \
        #        self.scores.unsqueeze(1).expand_as(word_probs)
        #    # Don't let EOS have children.
        #    for i in range(self.next_ys[-1].size(0)):
        #        if self.next_ys[-1][i] == self._eos:
        #            beam_scores[i] = -1e20

        #    # Block ngram repeats
        #    if self.block_ngram_repeat > 0:
        #        ngrams = []
        #        le = len(self.next_ys)
        #        for j in range(self.next_ys[-1].size(0)):
        #            hyp, _ = self.get_hyp(le-1, j)
        #            ngrams = set()
        #            fail = False
        #            gram = []
        #            for i in range(le-1):
        #                # Last n tokens, n = block_ngram_repeat
        #                gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
        #                # Skip the blocking if it is in the exclusion list
        #                if set(gram) & self.exclusion_tokens:
        #                    continue
        #                if tuple(gram) in ngrams:
        #                    fail = True
        #                ngrams.add(tuple(gram))
        #            if fail:
        #                beam_scores[j] = -10e20
        #else:
        #    beam_scores = word_probs[0]
        #print(beam_scores.numpy()[0], beam_scores.numpy()[0].sum(), )
        word_probs_smooth = word_probs.div(1)
        sm = self.s_func(word_probs_smooth)
        #print('s', sm.size())
        if prior is not None:
            sm_default = sm
            #print('p', prior.size())
            sm_prior = sm_default * prior
            sm_prior = sm_prior / torch.sum(sm_prior, -1, keepdim=True)
            sm_default_log = torch.log(sm_default)
            #torch.isinf in next version?
            sm_default_log[sm_default_log == float('-inf')] = 0
            sm_default_ent = -torch.sum( (sm_default * sm_default_log), 1, keepdim=True)
            #print('s1', sm_default_ent.size())
            #print('s2', sm_default.size())
            #print('s3', sm_prior.size())
            sm = torch.where(sm_default_ent < 2.7, sm_default, sm_prior)
            #print('r', res, res.size())
        #print(sm.numpy()[0], sm.numpy()[0].sum())
        rand_ys = sm.multinomial(1).view(-1)
        #print(rand_ys)
        #print(word_probs[rand_ys])
        scores_vec = np.array([float(sm[i][rand_ys[i]]) for i in range(self.batch_size)])
        scores_vec = np.log(scores_vec)
        scores_vec[list(self.finished)] = 0.
        #print(scores_vec)
        self.scores += scores_vec
        #print(self.scores)
        #flat_beam_scores = beam_scores.view(-1)
        #print('f', flat_beam_scores.numpy().shape)
        #best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
        #                                                    True, True)
        #print(best_scores.numpy(), best_scores_id.numpy())

        #self.all_scores.append(self.scores)
        #self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        #prev_k = best_scores_id / num_words
        #print('p', prev_k)
        #self.prev_ks.append(prev_k)
        #self.next_ys.append((best_scores_id - prev_k * num_words))
        self.next_ys.append((rand_ys))
        #self.rand_ys.append((rand_ys))
        #self.attn.append(attn_out.index_select(0, prev_k))
        #self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
        #        global_scores = self.global_scorer.score(self, self.scores)
        #        s = global_scores[i]
                #self.finished.append((s, len(self.next_ys) - 1, i))
                self.finished.add(i)
                #print(self.finished)

        # End condition is when top-of-beam is EOS and no global score.
        #if self.next_ys[-1][0] == self._eos:
        #    self.all_scores.append(self.scores)
        #    self.eos_top = True

    def done(self):
        return len(self.finished) >= self.batch_size
