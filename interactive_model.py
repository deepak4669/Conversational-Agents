#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interactive_model.py: Script to interact with already trained models.
Deepak Goyal<deepak4669@gmail.com>

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import operator
from queue import PriorityQueue

from collections import namedtuple, Counter

from Preprocessing.cornell_movie import normalizeString
from Preprocessing.cornell_movie import indexesFromSentence

from Models.transformer import subsequent_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(model, src, src_mask, max_len, start_symbol, trg=None):
    """ Greedy Decoding Algorithm

    @param model(torch model): The pytorch NLG model.
    @param src(Tensor of Tokens): The source sentence tokenised (normalised).
    @param src_mask(Tensor of Tokens): The mask is made such that it has 1 where src has non-zero values.
    @param max_len(Integer): The maximum length to be decoded.
    @param start_symbol(Integer): The start token.
    @param trg(Tensor): The target tensor used only while testing.

    @returns (Tensor,float): The decoded output from the model, The loss value.

    """

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    loss_val = 0

    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))

        prob = model.generator(out[:, -1])
        if trg is not None:
            loss_val += F.cross_entropy(prob, trg[0][i].view(-1)).item()
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word)], dim=1)
    return ys, loss_val / (max_len - 1)


class BeamNode:
    """
    Beam Node for the beam search decoding algorithm.
    """

    def __init__(self, previous_node, word_id, length, score):
        """ Init Beam Node

        @param previous_node(BeamNode): The previous node attached with current node.
        @param word_id
        @param length(Integer): The length of the output tokens upto this node.
        @param score(float): The cumulative score of the branch.
        """
        self.previous_node = previous_node
        self.word_id = word_id
        self.length = length
        self.score = score

    def eval(self):
        """
        @returns (float): The score of the branch normalised by its length.
        """
        return self.score / self.length


def beam_search(
        model,
        source,
        source_mask,
        max_len,
        start_symbol,
        beam_size,
        max_queue_size,
        num_sentences):
    """
    The Beam Search Decoding Algorithm

    @param model(pytorch model): The model for decoding next token.
    @param source(Tensor): The source sentence of tokens(normalised).
    @param source_mask(Tensor): The mask created for source tensor.(1 where non zero)
    @param max_len(Integer): The maximum length to be decoded.
    @param start_symbol(Integer): The start token.
    @param beam_size(Integer): The decoding parameter denoting number of candidates we keep for finall decoded solution.
    @param max_queue(Integer): The termination factor for the beam search. Number of nodes in th Priority Queue.
    @param num_sentence(Integer): The number of sentences to be decoded.

    @returns list[list[tokens]]: The list of decoded sentences.
    """

    encoder_outputs = model.encode(source, source_mask)

    decoder_input = torch.ones(1, 1).fill_(start_symbol).type_as(source.data)

    node = BeamNode(None, decoder_input, 1, 0)

    nodes = PriorityQueue()

    nodes.put((-node.eval(), node))
    queue_size = 1

    sentence_nodes = []

    while True:

        if queue_size > max_queue_size:
            break

        curr_score, curr_node = nodes.get()

        # decoder_input=curr_node.word_id

        if queue_size != 1:
            decoder_input = torch.cat([decoder_input, torch.ones(1, 1).type_as(
                source.data).fill_(curr_node.word_id.item())], dim=1)
        else:
            decoder_input = curr_node.word_id

        if curr_node.word_id.item() == 0 and curr_node.previous_node is not None:
            sentence_nodes.append((curr_score, curr_node))
            break

        decoder_output = model.decode(
            encoder_outputs,
            source_mask,
            Variable(decoder_input),
            Variable(
                subsequent_mask(
                    decoder_input.size(1)).type_as(
                    source.data)))

        prob = model.generator(decoder_output[:, -1])

        log_prob, indexes = torch.topk(prob, beam_size)
        nextnodes = []

        for i in range(beam_size):
            decoded_t = indexes[0][i]
            log_p = log_prob[0][i].item()

            node = BeamNode(
                curr_node,
                decoded_t,
                curr_node.length + 1,
                curr_node.score + log_p)
            score = -node.eval()
            nextnodes.append((score, node))

        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))

        queue_size += (len(nextnodes) - 1)

    if len(sentence_nodes) == 0:
        sentence_nodes = [nodes.get() for _ in range(numb_sentences)]

    utterances = []

    for score, n in sorted(sentence_nodes, key=operator.itemgetter(0)):
        curr_utterance = []
        curr_utterance.append(n.word_id)

        while n.previous_node is not None:
            n = n.previous_node
            curr_utterance.append(n.word_id)
        curr_utterance = curr_utterance[::-1]
        utterances.append(curr_utterance)

    return utterances


def top_k_top_p_filtering(
        logits,
        top_k=0,
        top_p=0.0,
        filter_value=-
        float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    @param logits(Tensor): logits distribution shape (vocabulary size)
    @param top_k >0(Integer): keep only top k tokens with highest probability (top-k filtering).
    @param top_p >0.0(Float): keep the top tokens with cumulative probability >= top_p (nucleus filtering).

    This code snippet is taken from Thom Wolf's Implementation.
    (https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)

    @returns (Tensor): Filtered logits.

    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the
        # top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the
        # threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[...,
                                                                :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def top_p_top_k(
        model,
        src,
        src_mask,
        max_len,
        start_symbol,
        temperature,
        top_k,
        top_p):
    """ The top-k top-p(Nucleus Sampling) decoding algorithm.

    @param model(pytorch model): The NLG model.
    @param src(Tensor): The source sentence vectorised and normalised.
    @param src_mask(Tensor): The source mask such that(1 where src is non-zero).
    @param max_len(Integer): The maximum length sentence to be decoded.
    @param start_symbol(Integer): The start token.
    @param temperature(float): The softmax temperature.
    @param top_k(Integer): The number of samples to be considered for top-k sampling.
    @param tok_p(float): The cumulative probability for the Nucleus Sampling.

    @returns (Tensor): Decoded tokens using top-k or top-p decoding algorithm.
    """

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    loss_val = 0

    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))

        logits = model.generator(out[:, -1])
        logits = logits / temperature

        filtered_logits = top_k_top_p_filtering(
            logits.view(-1), top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_word = torch.multinomial(probabilities, 1)

        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word.item())], dim=1)
    return ys


def evaluate(
        model,
        sentence,
        voc,
        max_length,
        decoding_algo,
        decoding_parameter):
    """ The function for generating decoded tokens.

    @param model(Pytorch model): The NLG model.
    @param sentence(str): The input sentence.
    @param voc(Vocabulary): The vocabulary considered for the model.
    @param max_length(Integer): The maximum length to be decoded.
    @param decoding_algo(str): The decoding algorithm to be used for decoding.
    @param decoding_parameter(float): The decoding parameter required by respective decoding algorithm.

    @returns list[str]: The decoded words.
    """

    tokenised_sentence = [indexesFromSentence(voc, sentence)]
    input_sentence = torch.LongTensor(tokenised_sentence)
    input_sentence_mask = (input_sentence != 0)

    input_sentence = input_sentence.view(-1, max_length + 2)
    input_sentence_mask = input_sentence_mask.view(1, -1, max_length + 2)

    output_sentence = ''
    if decoding_algo == 'greedy':
        output_sentence, _ = greedy_decode(model, input_sentence.to(
            device), input_sentence_mask.to(device), max_length, 1)
    elif decoding_algo == 'beam-search':
        output_sentence = torch.LongTensor(
            beam_search(
                model,
                input_sentence.to(device),
                input_sentence_mask.to(device),
                10,
                1,
                int(decoding_parameter),
                2000,
                1)[0])
    elif decoding_algo == 'top-k':
        output_sentence = top_p_top_k(
            model,
            input_sentence.to(device),
            input_sentence_mask.to(device),
            max_length,
            1,
            1.0,
            int(decoding_parameter),
            0)
    else:
        output_sentence = top_p_top_k(
            model,
            input_sentence.to(device),
            input_sentence_mask.to(device),
            max_length,
            1,
            1.0,
            0,
            decoding_parameter)

    output_sentence = output_sentence.view(-1)

    decoded_words = [voc.index2word[id.item()] for id in output_sentence]

    return decoded_words


def evaluateInput(model, voc, max_length, decoding_algo, decoding_parameter):
    """ The Interactive input-output conversation function.

    @param model(Pytorch model): The pytorch model to be used for decoding.
    @param voc(Vocabulary): The vocabulary used for decoding.
    @param max_length(Integer): The maximum length of the decoding to be used.
    @param decoding_algo(str): The decoding algo to be used.
    @param decoding_parameter(float): The required parameter for decoding.



    """

    input_sentence = ''

    while(1):
        try:
            input_sentence = input("Human: ")
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            input_sentence = normalizeString(input_sentence)

            output_words = evaluate(
                model,
                input_sentence,
                voc,
                max_length,
                decoding_algo,
                decoding_parameter)
            output_words[:] = [
                x for x in output_words if not(
                    x == "SOS" or x == "EOS" or x == "PAD")]

            print("Machine:", " ".join(output_words))

        except KeyError:
            print("Unkown Word..!!")


def interact(working_model, decoding_algo, decoding_parameter, voc):
    """ The method to load checkpoints and perform interaction.

    @param working_model(pytorch model): The model to be used.
    @param decoding_algo(str): The decoding algorithm to be used.
    @param decoding_parameter(float): The decoding parameter required.
    @param voc(Vocabulary): The voc to be used.


    """

    loadFile = 'C:\\Users\\deepa\\Final Year Project\\Model Data\\transformer\\cornell-movie-40k\\200_checkpoint.tar'
    if(loadFile):
        checkpoint = torch.load(loadFile, map_location='cpu')
        working_model.load_state_dict(checkpoint['model'])

    working_model.eval()
    evaluateInput(working_model, voc, 10, decoding_algo, decoding_parameter)
