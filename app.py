import gradio as gr
import os
from pinecone_integration import PineconeIndex
from qa_model import QAModel


PI = PineconeIndex()
PI.build_index()
qamodel = QAModel()
model, tokenizer = qamodel.load_sharded_model()

def request_answer(query):
    search_results = PI.search(query)
    answers = []
    for r in search_results['matches']:
        if r['score'] >= 0.45:
            tokenized_context = tokenizer(r['metadata']['text'])
            query_to_model = """You are doctor who knows cures to diseases. If you don't know, say you don't know. Please respond appropriately based on the context provided.\n\nContext: {}\n\n\nResponse: """
            for ind in range(0, len(tokenized_context['input_ids']), 512-42):                        
                decoded_tokens_for_context = tokenizer.batch_decode([tokenized_context['input_ids'][ind:ind+470]], skip_special_tokens=True)
                response = qamodel.query_model(model, tokenizer, query_to_model.format(decoded_tokens_for_context[0]))
                
                if not "don't know" in response:
                    answers.append(response)

    if len(answers) == 0:
        return 'Not enough information to answer the question'
    return '\n'.join(answers)


demo = gr.Interface(
    fn=request_answer,
    inputs=[
        gr.components.Textbox(label="User question(Response may take up to 2 mins because of hardware limitation)"),
    ],
    outputs=[
        gr.components.Textbox(label="Output (The answer is meant as a reference and not actual advice)"),
    ],
    cache_examples=True,
    title="MedQA assistant",
    description='Check out the repository at: https://github.com/anandshah98/MedQA',
)

demo.launch()
