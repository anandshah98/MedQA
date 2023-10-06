import gradio as gr
import os
from pinecone_integration import PineconeIndex
from qa_model import QAModel


PI = PineconeIndex()
PI.build_index()
qamodel = QAModel()
model, tokenizer = qamodel.load_sharded_model()