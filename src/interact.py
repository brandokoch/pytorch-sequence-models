import torch 
import model_dispatcher
import config
import json
import tensorflow as tf
import os
from keras_preprocessing.text import tokenizer_from_json

def load_model():
    model=model_dispatcher.models[config.MODEL].to('cuda')
    model_dict_pth=os.path.join('runs',config.RUN_NAME,config.MODEL+'.pt')
    model.load_state_dict(torch.load(model_dict_pth, map_location=torch.device('cuda')))
    model.eval()
    return model

def get_preprocess_funcs():

    if 'sentimentclf' in config.MODEL:

        tok_pth=os.path.join('runs',config.RUN_NAME, config.MODEL+'_tok.json')
        with open(tok_pth) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)

        def preprocess_sent_clf(input_str):
            x=[input_str]
            x_pro=tokenizer.texts_to_sequences(x)
            x_pro=tf.keras.preprocessing.sequence.pad_sequences(x_pro, maxlen=128)
            x_pro=torch.tensor(x_pro, device=torch.device('cuda'))
            return x_pro

        def postprocess_sent_clf(out):
            return torch.sigmoid(out).detach().cpu().numpy()

        return preprocess_sent_clf, postprocess_sent_clf

    if 'languagemodel' in config.MODEL:

        tok_pth=os.path.join('runs',config.RUN_NAME, config.MODEL+'_tok.json')
        with open(tok_pth) as f:
            word2idx = json.load(f)
            idx2word= {v: k for k, v in word2idx.items()}

        def preprocess_LM(input_str):
            x_pro=[word2idx[w] for w in input_str.split(' ') if w != '']

            x_pro=[x_pro]
            x_pro=torch.tensor(x_pro, device=torch.device('cuda'))
            return x_pro

        def postprocess_LM(out):
            out=out.detach().cpu()
            out=out[:,:,-1]

            out=torch.softmax(out, dim=1)
            out=torch.argmax(out)
            
            out=idx2word[int(out.numpy())]
            return out

        return preprocess_LM, postprocess_LM
        


if __name__=="__main__":

    preprocessor,postprocessor=get_preprocess_funcs()
    model=load_model()

    while True:
        input_str=input("Enter input: ")

        with torch.no_grad():
            x_pro=preprocessor(input_str)
            out=model(x_pro)
            out=postprocessor(out)
            print(out)
    
