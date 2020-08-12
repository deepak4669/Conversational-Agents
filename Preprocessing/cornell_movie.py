#
# This code takes borrow heavily from chatbot tutorial presented over pytorch website
#

# necessary imports

import os
import re
import torch
import random
import itertools
import time


PAD_Token = 0
START_Token = 1
END_Token = 2


def get_lines_conversations(data_folder):
    """
    Loads movie lines and conversations from the dataset.
    
    data_folder: Destination where conversations and lines are stored.
    
    movie_lines: Consist of movie lines as given by the dataset.
    movie_conversations: Consist of movie conversations as given by the dataset.
    
    """
    movie_lines = []
    movie_conversations = []

    with open(
        os.path.join(data_folder, "movie_lines.txt"), "r", encoding="iso-8859-1"
    ) as f:
        for line in f:
            movie_lines.append(line)

    with open(
        os.path.join(data_folder, "movie_conversations.txt"), "r", encoding="iso-8859-1"
    ) as f:
        for line in f:
            movie_conversations.append(line)

    return movie_lines, movie_conversations


def loadLines(movie_lines, fields, exceptions):
    lines = {}
    for lineid in range(len(movie_lines)):

        line = movie_lines[lineid]
        values = line.split(" +++$+++ ")

        lineVals = {}

        # print("values"+str(len(values)))
        # print("fields"+str(len(fields)))

        for i, field in enumerate(fields):
            try:
                lineVals[field] = values[i]
            except:
                print("Exception: " + str(len(values)))
                exceptions.append(lineid)

        lines[lineVals["lineID"]] = lineVals

    return lines


def loadConversations(movie_conversations, lines, fields):
    conversations = []

    for convo in movie_conversations:
        values = convo.split(" +++$+++ ")
        conVals = {}

        for i, field in enumerate(fields):
            conVals[field] = values[i]

        lineIDs = eval(conVals["utteranceIDs"])

        conVals["lines"] = []

        for lineID in lineIDs:
            conVals["lines"].append(lines[lineID])
        conversations.append(conVals)

    return conversations


def sentencePairs(conversations):
    qr_pairs = []

    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            query = conversation["lines"][i]["text"].strip()
            response = conversation["lines"][i + 1]["text"].strip()

            if query and response:
                qr_pairs.append([query, response])

    return qr_pairs


class Vocabulary:
    def __init__(self):
        self.trimmed = False
        self.word2count = {}
        self.index2word = {PAD_Token: "PAD", START_Token: "SOS", END_Token: "EOS"}
        self.word2index = {"PAD": PAD_Token, "SOS": START_Token, "EOS": END_Token}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words = self.num_words + 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):

        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for word, freq in self.word2count.items():
            if freq >= min_count:
                keep_words.append(word)

        self.word2count = {}
        self.index2word = {PAD_Token: "PAD", START_Token: "SOS", END_Token: "EOS"}
        self.word2index = {"PAD": PAD_Token, "SOS": START_Token, "EOS": END_Token}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


Max_Length = 10


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def readVocs(qr_pairs):

    for qr_pair in qr_pairs:
        qr_pair[0] = normalizeString(qr_pair[0])
        qr_pair[1] = normalizeString(qr_pair[1])

    voc = Vocabulary()
    return voc, qr_pairs


def filterPair(pair):
    return len(pair[0].split(" ")) < Max_Length and len(pair[1].split(" ")) < Max_Length


def filterPairs(qr_pairs):
    return [pair for pair in qr_pairs if filterPair(pair)]


def prepareDataset(qr_pairs):
    voc, qr_pairs = readVocs(qr_pairs)
    qr_pairs = filterPairs(qr_pairs)

    for pair in qr_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    #     print("Number"+str(voc.num_words))
    return voc, qr_pairs


Min_Count = 3


def trimRareWords(voc, qr_pairs):

    voc.trim(Min_Count)
    keep_pairs = []

    for pair in qr_pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep_input = True
        keep_output = True

        for word in input_sentence.split(" "):
            if word not in voc.word2index:
                keep_input = False
                break

        for word in output_sentence.split(" "):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    return keep_pairs


def indexesFromSentence(voc, sentence):
    tokenised_sentence = []
    tokenised_sentence.append(START_Token)

    for word in sentence.split(" "):
        tokenised_sentence.append(voc.word2index[word])

    tokenised_sentence.append(END_Token)

    assert len(tokenised_sentence) <= Max_Length + 2
    for _ in range(Max_Length + 2 - len(tokenised_sentence)):
        tokenised_sentence.append(PAD_Token)

    return tokenised_sentence


def binaryMatrix(l, value=PAD_Token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)

    return m


def inputVar(voc, l):

    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    input_lengths = torch.tensor([len(index) for index in indexes_batch])
    padVar = torch.LongTensor(indexes_batch)
    return input_lengths, padVar


def outputVar(voc, l):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = torch.tensor([len(index) for index in indexes_batch])
    mask = binaryMatrix(indexes_batch)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(indexes_batch)
    return max_target_len, mask, padVar


def batch2TrainData(voc, pair_batch):
    # sort function see
    input_batch = []
    output_batch = []

    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    input_lengths, tokenised_input = inputVar(voc, input_batch)
    max_out_length, mask, tokenised_output = outputVar(voc, output_batch)
    return input_lengths, tokenised_input, max_out_length, mask, tokenised_output


# print("Number of query-response pairs after all the preprocessing: "+str(len(pairs)))

# #Sample batch
# batch=[random.choice(pairs) for _ in range(5)]
# input_lengths,tokenised_input,max_out_length,mask,tokenised_output=batch2TrainData(voc,batch)

# print("Input length: "+str(input_lengths)+" Size: "+str(input_lengths.shape))
# print("-"*80)
# print("Tokenised Input: "+str(tokenised_input)+" Size: "+str(tokenised_input.shape))
# print("-"*80)
# print("Max out length: "+str(max_out_length)+" Size: "+str(max_out_length.shape))
# print("-"*80)
# print("Mask: "+str(mask)+" Size: "+str(mask.shape))
# print("-"*80)
# print("Tokenised Output: "+str(tokenised_output)+" Size: "+str(tokenised_output.shape))
# print("-"*80)


def get_vocabulary():
    path = "C:\\Users\\deepa\\Conversational Agents\\Datasets"
    dataset = "cornell movie-dialogs corpus"

    data_folder = os.path.join(path, dataset)

    print("The final data corpus folder: " + str(data_folder))
    print("Extracting movie lines and movie conversations...")

    movie_lines, movie_conversations = get_lines_conversations(data_folder)

    print("Number of distinct lines: " + str(len(movie_lines)))
    print("Number of conversations: " + str(len(movie_conversations)))
    print(
        "Average Number of lines per conversations: "
        + str(len(movie_lines) / len(movie_conversations))
    )

    lines = {}
    conversations = []
    qr_pairs = []

    exceptions = []

    PAD_Token = 0
    START_Token = 1
    END_Token = 2

    movie_lines_fields = ["lineID", "characterID", "movieID", "character", "text"]
    movie_convo_fields = ["charcaterID", "character2ID", "movieID", "utteranceIDs"]

    lines = loadLines(movie_lines, movie_lines_fields, exceptions)
    conversations = loadConversations(movie_conversations, lines, movie_convo_fields)
    qr_pairs = sentencePairs(conversations)

    voc, pairs = prepareDataset(qr_pairs)

    pairs = trimRareWords(voc, pairs)

    return voc


voc = get_vocabulary()
