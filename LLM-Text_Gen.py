import logging
import sys
from aitextgen import aitextgen
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def Train_Model(file_name):
    logging.basicConfig(format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO)

    ai = aitextgen(model="EleutherAI/gpt-neo-125M", to_gpu=False)
    ai.train(file_name,line_by_line=False,from_cache=False,num_steps=3000,generate_every=1000,save_every=1000,save_gdrive=False,learning_rate=1e-3,fp16=False,batch_size=1)
    ai.save_model("trained_model")

def Generate_Text_with_Trained_Model(Prompt,Decoding_method):
    ai = aitextgen(model_folder="trained_model", to_gpu=False)
    


    #decoding Methods
    if(Decoding_method == "Greedy Search"):
        text = ai.generate(n=1, prompt=Prompt, max_length=50, return_as_list=True, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0)
    elif (Decoding_method == "Beam Search"):
        text = ai.generate(n=1, prompt=Prompt, max_length=50, return_as_list=True, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    elif (Decoding_method == "Top-K"):
        text = ai.generate(n=1, prompt=Prompt, max_length=50, return_as_list=True, temperature=0.7, top_k=50, top_p=0.0, repetition_penalty=1.0)
    elif (Decoding_method == "Top-P"):
         text = ai.generate(n=1, prompt=Prompt, max_length=50, return_as_list=True, temperature=0.7, top_k=0, top_p=0.92, repetition_penalty=1.0)
    else:
        print("Unable to execute chosen decoding method.")
        sys.exit("exiting program")

    print(Decoding_method+" Sampling Output:\n" + 100 * "-")
    print(text[0])

def main():
    file = input("Enter file name you are using to train model\n")
    Train_Model(file)

    prompt = input("Enter prompt for text generation\n")
    Decoding = input("Choose Decoding method from:\n Greedy Search\n Beam Search\n Top-K\n Top-P\n")

    Generate_Text_with_Trained_Model(prompt,Decoding)

if __name__ == '__main__':
    main()