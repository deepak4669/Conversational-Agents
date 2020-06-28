#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
For training and interacting with already trained models.
Deepak Goyal <deepak4669@gmail.com>

Usage:
    python run.py [action][model][decoding algorithm][decoding parameter]

action:
    transformer              Use Transformer model
decoding algorithm:
    greedy                   Use Greedy Decoding algorithm for decoding the model
    beam-search              Use Beam Search decoding algorithm for decoding the model
    top-k                    Use top-k decoding algorithm for decoding the model
    nucleus                  Use Nucleus Sampling decoding algoritm for decoding the model
decoding parameter:
    Any Real                 For greedy decoding algorithm there is no need of specific decoding parameter however it is required you enter something.
    beam-width               The top candidates that we keep as our final decoded tokens.(positive integer)
    k                        The top k candidates from which we randomly sample the tokens.(positive integer)
    p                        The cumulative probabilty to be considered while collecting potential tokens in Nuclues Sampling.

"""


from Preprocessing.cornell_movie import get_vocabulary
import sys
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arguments = sys.argv
# Deleting the first command line argument which is run.py itself.
del arguments[0:1]
action, model, decoding_algo, decoding_parameter = arguments

voc = get_vocabulary()

if action == 'interact':

    if model == 'transformer':
        from Models.transformer import make_model
        working_model = make_model(
            voc.num_words, voc.num_words, 2, 256, 512, 8, 0.1)
        working_model.to(device)
        print(working_model)

        from interactive_model import interact

        interact(working_model, decoding_algo, float(decoding_parameter), voc)

    # if model=='gru':

    #     from Models.seq2seq_attn import make_model
    #     working_model=make_model(voc.num_words,500,2,2,0.1,0.1,'dot')
    #     working_model.to(device)
    #     from interactive_model import interact
    #     interact
