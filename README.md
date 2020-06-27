# Conversational-Agents
The repository contains code for final year project titled "Making Machines Converse Better". The project focusses on Open-Domain Conversational Agents, the thesis for the same can be found [here](https://drive.google.com/file/d/19Gs8X_4BFzuuV2Yk4njb0myq1ZOZMfxF/view?usp=sharing).

Pretrained weights for the models can be downloaded from [here](https://www.dropbox.com/sh/ojl5bh5uwz2smr2/AACahGWXQNyE1oQRLqF0_N11a?dl=0).

## Instructions for running interactive model

1. Install all dependencies using requirements.txt.
2. Download the pre-trained weights from the above link and place them in any directory but change the same in interactive_model.py interact function.
3. Run the run.py the command line arguments are in this manner [action],[model],[decoding algorithm],[decoding algorithm parameter]. Example: For an interactive model with tranformer architecture decoded using greedy decoding algorithm the commmand will be:

```
python run.py interact transformer greedy
```
