import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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


def evaluate(model,sentence,voc,max_length):

    tokenised_sentence=[indexesFromSentence(voc,sentence)]
    input_sentence=torch.LongTensor(tokenised_sentence)
    input_sentence_mask=(input_sentence!=0)

    input_sentence=input_sentence.view(-1,max_length+2)
    input_sentence_mask=input_sentence_mask.view(1,-1,max_length+2)

    output_sentence,_=greedy_decode(model,input_sentence.to(device),input_sentence_mask.to(device),max_length,1)
    # output_sentence=torch.LongTensor(beam_search(model,input_sentence.to(device),input_sentence_mask.to(device),10,1,20,2000,1)[0])
    # output_sentence=top_p_top_k(model,input_sentence.to(device),input_sentence_mask.to(device),max_length,1,1.0,0,0.5)

    output_sentence=output_sentence.view(-1)

    decoded_words=[voc.index2word[id.item()] for id in output_sentence]

    return decoded_words

def evaluateInput(model,voc,max_length):
    input_sentence=''

    while(1):
        try:
            input_sentence=input("Human: ")
            if input_sentence=='q' or input_sentence=='quit':
                break
            input_sentence=normalizeString(input_sentence)

            output_words=evaluate(model,input_sentence,voc,max_length)
            output_words[:]=[x for x in output_words if not(x=="SOS" or x=="EOS" or x=="PAD")]

            print("Machine:"," ".join(output_words))

        except KeyError:
            print("Unkown Word..!!")




def interact(working_model,decoding_algo,voc):
    
    loadFile='C:\\Users\\deepa\\Final Year Project\\Model Data\\transformer\\cornell-movie-40k\\200_checkpoint.tar'
    if(loadFile):
        checkpoint=torch.load(loadFile, map_location='cpu')
        working_model.load_state_dict(checkpoint['model'])

    working_model.eval()
    evaluateInput(working_model,voc,10)



