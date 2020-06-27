import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from queue import PriorityQueue
import operator

from collections import namedtuple, Counter

from Preprocessing.cornell_movie import normalizeString
from Preprocessing.cornell_movie import indexesFromSentence

from Models.transformer import subsequent_mask


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decode(model, src, src_mask,max_len, start_symbol,trg=None):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    loss_val=0

    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        
        prob = model.generator(out[:, -1])
        if trg!=None:
            loss_val+=F.cross_entropy(prob,trg[0][i].view(-1)).item()
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys,loss_val/(max_len-1)



class BeamNode:

    def __init__(self,previous_node,word_id,length,score):
        
        self.previous_node=previous_node
        self.word_id=word_id
        self.length=length
        self.score=score

    def eval(self):
        return self.score/self.length


def beam_search(model,source,source_mask,max_len,start_symbol,beam_size,max_queue_size,num_sentences):

    encoder_outputs=model.encode(source,source_mask)

    decoder_input=torch.ones(1,1).fill_(start_symbol).type_as(source.data)


    node=BeamNode(None,decoder_input,1,0)

    nodes=PriorityQueue()

    nodes.put((-node.eval(),node))
    queue_size=1

    sentence_nodes=[]

    while True:

        if queue_size>max_queue_size:
            break

        curr_score,curr_node=nodes.get()

        # decoder_input=curr_node.word_id

        if queue_size!=1:
            decoder_input=torch.cat([decoder_input,torch.ones(1, 1).type_as(source.data).fill_(curr_node.word_id.item())], dim=1)
        else:
            decoder_input=curr_node.word_id

        if curr_node.word_id.item()==0 and curr_node.previous_node!=None:
            sentence_nodes.append((curr_score,curr_node))
            break


        decoder_output=model.decode(encoder_outputs,source_mask,Variable(decoder_input),Variable(subsequent_mask(decoder_input.size(1)).type_as(source.data)))

        prob=model.generator(decoder_output[:,-1])

        log_prob,indexes=torch.topk(prob,beam_size)
        nextnodes=[]

        for i in range(beam_size):
            decoded_t=indexes[0][i]
            log_p=log_prob[0][i].item()

            node=BeamNode(curr_node,decoded_t,curr_node.length+1,curr_node.score+log_p)
            score=-node.eval()
            nextnodes.append((score,node))
        
        for i in range(len(nextnodes)):
            score,nn=nextnodes[i]
            nodes.put((score,nn))

        queue_size+=(len(nextnodes)-1)
    
    if len(sentence_nodes)==0:
        sentence_nodes=[nodes.get() for _ in range(numb_sentences)]
    

    utterances=[]

    for score,n in sorted(sentence_nodes,key=operator.itemgetter(0)):
        curr_utterance=[]
        curr_utterance.append(n.word_id)

        while n.previous_node!=None:
            n=n.previous_node
            curr_utterance.append(n.word_id)
        curr_utterance=curr_utterance[::-1]
        utterances.append(curr_utterance)

    return utterances


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def top_p_top_k(model, src, src_mask,max_len, start_symbol,temperature,top_k,top_p):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    loss_val=0

    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        
        logits = model.generator(out[:, -1])
        logits=logits/temperature

        filtered_logits = top_k_top_p_filtering(logits.view(-1), top_k=top_k, top_p=top_p)
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_word = torch.multinomial(probabilities, 1)
        
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word.item())], dim=1)
    return ys



def evaluate(model,sentence,voc,max_length,decoding_algo,decoding_parameter):

    tokenised_sentence=[indexesFromSentence(voc,sentence)]
    input_sentence=torch.LongTensor(tokenised_sentence)
    input_sentence_mask=(input_sentence!=0)

    input_sentence=input_sentence.view(-1,max_length+2)
    input_sentence_mask=input_sentence_mask.view(1,-1,max_length+2)

    output_sentence=''
    if decoding_algo=='greedy':
        output_sentence,_=greedy_decode(model,input_sentence.to(device),input_sentence_mask.to(device),max_length,1)
    elif decoding_algo=='beam-search':
        output_sentence=torch.LongTensor(beam_search(model,input_sentence.to(device),input_sentence_mask.to(device),10,1,int(decoding_parameter),2000,1)[0])
    elif decoding_algo=='top-k':
        output_sentence=top_p_top_k(model,input_sentence.to(device),input_sentence_mask.to(device),max_length,1,1.0,int(decoding_parameter),0)
    else:
        output_sentence=top_p_top_k(model,input_sentence.to(device),input_sentence_mask.to(device),max_length,1,1.0,0,decoding_parameter)

    output_sentence=output_sentence.view(-1)

    decoded_words=[voc.index2word[id.item()] for id in output_sentence]

    return decoded_words

def evaluateInput(model,voc,max_length,decoding_algo,decoding_parameter):
    input_sentence=''

    while(1):
        try:
            input_sentence=input("Human: ")
            if input_sentence=='q' or input_sentence=='quit':
                break
            input_sentence=normalizeString(input_sentence)

            output_words=evaluate(model,input_sentence,voc,max_length,decoding_algo,decoding_parameter)
            output_words[:]=[x for x in output_words if not(x=="SOS" or x=="EOS" or x=="PAD")]

            print("Machine:"," ".join(output_words))

        except KeyError:
            print("Unkown Word..!!")




def interact(working_model,decoding_algo,decoding_parameter,voc):
    
    loadFile='C:\\Users\\deepa\\Final Year Project\\Model Data\\transformer\\cornell-movie-40k\\200_checkpoint.tar'
    if(loadFile):
        checkpoint=torch.load(loadFile, map_location='cpu')
        working_model.load_state_dict(checkpoint['model'])

    working_model.eval()
    evaluateInput(working_model,voc,10,decoding_algo,decoding_parameter)



