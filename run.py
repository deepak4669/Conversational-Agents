import sys
import torch

arguments=sys.argv
del arguments[0:1]

action,model,decoding_algo=arguments

from Preprocessing.cornell_movie import get_vocabulary


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
voc=get_vocabulary()

if action=='interact':
    
    if model=='transformer':
        from Models.transformer import make_model
        working_model=make_model(voc.num_words,voc.num_words,2,256,512,8,0.1)
        working_model.to(device)
        print(working_model)

        from interactive_model import interact

        interact(working_model,"greedy",voc)











