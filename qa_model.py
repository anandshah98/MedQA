import os
import torch

from transformers import AutoModel, AutoConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

class QAModel():
    def __init__(self, checkpoint="google/flan-t5-xl"):
        self.checkpoint = checkpoint
        self.tmpdir = f"{self.checkpoint.split('/')[-1]}-sharded"

    def store_sharded_model(self):
        tmpdir = self.tmpdir
        
        checkpoint = self.checkpoint
        
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
            print(f"Directory created - {tmpdir}")
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            print(f"Model loaded - {checkpoint}")
            model.save_pretrained(tmpdir, max_shard_size="200MB")

    def load_sharded_model(self):
        tmpdir = self.tmpdir
        if not os.path.exists(tmpdir):
            self.store_sharded_model()
            
        checkpoint = self.checkpoint
        

        config = AutoConfig.from_pretrained(checkpoint)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        with init_empty_weights():
            model = AutoModelForSeq2SeqLM.from_config(config)
            # model = AutoModelForSeq2SeqLM.from_pretrained(tmpdir)

        model = load_checkpoint_and_dispatch(model, checkpoint=tmpdir, device_map="auto")
        return model, tokenizer

    def query_model(self, model, tokenizer, query):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return tokenizer.batch_decode(model.generate(**tokenizer(query, return_tensors='pt').to(device)), skip_special_tokens=True)[0]