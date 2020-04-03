import torch
import torch.nn.functional as F

from onmt.translate.decode_strategy import DecodeStrategy
from onmt.utils.misc import tile


# def sample_with_temperature(logits, sampling_temp, keep_topk):
#     """Select next tokens randomly from the top k possible next tokens.

#     Samples from a categorical distribution over the ``keep_topk`` words using
#     the category probabilities ``logits / sampling_temp``.

#     Args:
#         logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
#             These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
#             (The distribution actually uses the log-probabilities
#             ``logits - logits.logsumexp(-1)``, which equals the logits if
#             they are log-probabilities summing to 1.)
#         sampling_temp (float): Used to scale down logits. The higher the
#             value, the more likely it is that a non-max word will be
#             sampled.
#         keep_topk (int): This many words could potentially be chosen. The
#             other logits are set to have probability 0.

#     Returns:
#         (LongTensor, FloatTensor):

#         * topk_ids: Shaped ``(batch_size, 1)``. These are
#           the sampled word indices in the output vocab.
#         * topk_scores: Shaped ``(batch_size, 1)``. These
#           are essentially ``(logits / sampling_temp)[topk_ids]``.
#     """

#     if sampling_temp == 0.0 or keep_topk == 1:
#         # For temp=0.0, take the argmax to avoid divide-by-zero errors.
#         # keep_topk=1 is also equivalent to argmax.
#         topk_scores, topk_ids = logits.topk(1, dim=-1)
#         if sampling_temp > 0:
#             topk_scores /= sampling_temp
#     else:
#         logits = torch.div(logits, sampling_temp)

#         if keep_topk > 0:
#             top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
#             kth_best = top_values[:, -1].view([-1, 1])
#             kth_best = kth_best.repeat([1, logits.shape[1]]).float()

#             # Set all logits that are not in the top-k to -10000.
#             # This puts the probabilities close to 0.
#             ignore = torch.lt(logits, kth_best)
#             logits = logits.masked_fill(ignore, -10000)

#         dist = torch.distributions.Multinomial(
#             logits=logits, total_count=1)
#         topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
#         topk_scores = logits.gather(dim=1, index=topk_ids)
#     return topk_ids, topk_scores

def sample_with_temperature_default_logprob(logits, logits_default, sampling_temp, keep_topk):
    """Select next tokens randomly from the top k possible next tokens.
    
    TVDC NOTE: priors mess with log_probabilities; this function
    adapts sample function in order to take global score from decoder
    into account that uses logprobs without prior adaptation


    Samples from a categorical distribution over the ``keep_topk`` words using
    the category probabilities ``logits / sampling_temp``.

    Args:
        logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        sampling_temp (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        keep_topk (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.

    Returns:
        (LongTensor, FloatTensor):

        * topk_ids: Shaped ``(batch_size, 1)``. These are
          the sampled word indices in the output vocab.
        * topk_scores: Shaped ``(batch_size, 1)``. These
          are essentially ``(logits / sampling_temp)[topk_ids]``.

    """

    if sampling_temp == 0.0 or keep_topk == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # keep_topk=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if sampling_temp > 0:
            topk_scores /= sampling_temp
    else:
        logits = torch.div(logits, sampling_temp)

        if keep_topk > 0:
            top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
            kth_best = top_values[:, -1].view([-1, 1])
            kth_best = kth_best.repeat([1, logits.shape[1]]).float()

            # Set all logits that are not in the top-k to -10000.
            # This puts the probabilities close to 0.
            ignore = torch.lt(logits, kth_best)
            logits = logits.masked_fill(ignore, -10000)

        dist = torch.distributions.Multinomial(
            logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits_default.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores

class PriorSampling(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    The ``scores`` attribute's lists are the score, after applying temperature,
    of the final prediction (either EOS or the final token in the event
    that ``max_length`` is reached)

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        batch_size (int): See base.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        sampling_temp (float): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        keep_topk (int): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
    """

    def __init__(self, pad, bos, eos, batch_size, sample_size, min_length,
                 block_ngram_repeat, exclusion_tokens, return_attention,
                 max_length, sampling_temp, keep_topk):
        #assert block_ngram_repeat == 0
        super(PriorSampling, self).__init__(
            pad, bos, eos, batch_size, sample_size, min_length, block_ngram_repeat,
            exclusion_tokens, return_attention, max_length)
        self.sampling_temp = sampling_temp
        self.keep_topk = keep_topk
        self.topk_scores = None
        self.sample_size = sample_size
        self.block_ngram_repeat = block_ngram_repeat

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None):
        """Initialize for decoding."""
        #fn_map_state = None
        def fn_map_state(state, dim):
            return tile(state, self.sample_size, dim=dim)

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, self.sample_size, dim=1)
                                for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, self.sample_size, dim=1)
            mb_device = memory_bank.device
        if src_map is not None:
            src_map = tile(src_map, self.sample_size, dim=1)
        if device is None:
            device = mb_device

        self.memory_lengths = tile(src_lengths, self.sample_size)
        super(PriorSampling, self).initialize(
            memory_bank, self.memory_lengths, src_map, device)
        self.select_indices = torch.arange(
            self.batch_size * self.sample_size, dtype=torch.long, device=device)
        self.original_batch_idx = tile(torch.arange(
            self.batch_size, dtype=torch.long, device=device), self.sample_size)
        return fn_map_state, memory_bank, self.memory_lengths, src_map

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def batch_offset(self):
        return self.select_indices

    def advance(self, log_probs, attn, prior=None):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (FloatTensor): Shaped ``(1, B, inp_seq_len)``.
        """

        self.ensure_min_length(log_probs)
        if self.block_ngram_repeat:
            self.block_ngram_repeats(log_probs)
        ## don't want no unk generation
        log_probs[:,0] = -1e20
        default_log_probs = log_probs
        if prior is not None:
            prior_log_probs = log_probs + prior.log()
            default_probs = F.softmax(log_probs, dim=1)
            default_probs_log = default_probs.log()
            default_probs_log[default_probs_log == float('-inf')] = -1e20
            vector_entropies = -torch.sum(default_probs * default_probs_log, 1, keepdim=True)
            #hard-coded entropy cutoff; this needs to change
            log_probs = torch.where(vector_entropies < 4, default_log_probs, prior_log_probs) #4.5
            
        topk_ids, self.topk_scores = sample_with_temperature_default_logprob(
            log_probs, default_log_probs, self.sampling_temp, self.keep_topk)
        self.is_finished = topk_ids.eq(self.eos)
        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 0)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        # shape: (sum(~ self.is_finished), 1)
        finished_batches = self.is_finished.view(-1).nonzero()
        for b in finished_batches.view(-1):
            b_orig = self.original_batch_idx[b]
            self.scores[b_orig].append(self.topk_scores[b, 0])
            self.predictions[b_orig].append(self.alive_seq[b, 1:])
            self.attention[b_orig].append(
                self.alive_attn[:, b, :self.memory_lengths[b]]
                if self.alive_attn is not None else [])
        self.done = self.is_finished.all()
        if self.done:
            return
        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        if self.alive_attn is not None:
            self.alive_attn = self.alive_attn[:, is_alive]
        self.select_indices = is_alive.nonzero().view(-1)
        self.original_batch_idx = self.original_batch_idx[is_alive]
