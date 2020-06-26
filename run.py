import sys

arguments=sys.argv
del arguments[0:1]

action,model,decoding_algo=arguments

from preprocessing.cornell-movie import get_vocabulary

voc=get_vocabulary()

if action=='interact':
    working_model=_
    if model=='transformer':
        from Models.transformer import make_model
        working_model=make_model







