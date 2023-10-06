import pandas as pd
import os
from tqdm.auto import tqdm
import pinecone
from sentence_transformers import SentenceTransformer
import torch

class PineconeIndex:

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.sm = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.index_name = 'semantic-search-fast-med'
        self.index = None

    def init_pinecone(self):

        index_name = self.index_name
        sentence_model = self.sm

        # get api key from app.pinecone.io
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'None')
        # find your environment next to the api key in pinecone console
        PINECONE_ENV = os.environ.get('PINECONE_ENV', "us-west4-gcp")

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

#         pinecone.delete_index(index_name)

        # only create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=sentence_model.get_sentence_embedding_dimension(),
                metric='cosine'
            )

        # now connect to the index
        self.index = pinecone.GRPCIndex(index_name)
        return self.index

    def build_index(self):

        if self.index is None:
            index = self.init_pinecone()
        else:
            index = self.index

        if index.describe_index_stats()['total_vector_count']:
            "Index already built"
            return

        sentence_model = self.sm

        x = pd.read_excel('/kaggle/input/drug-p/Diseases_data_W.xlsx')

        question_dict = {'About': 'What is {}?', 'Symptoms': 'What are symptoms of {}?',
                         'Causes': 'What are causes of {}?',
                         'Diagnosis': 'What are diagnosis for {}?', 'Risk Factors': 'What are the risk factors for {}?',
                         'Treatment Options': 'What are the treatment options for {}?',
                         'Prognosis and Complications': 'What are the prognosis and complications?'}
        context = []
        disease_list = []

        for i in range(len(x)):
            disease = x.iloc[i, 0]
            if disease.strip().lower() in disease_list:
                continue

            disease_list.append(disease.strip().lower())

            conditions = x.iloc[i, 1:].dropna().index
            answers = x.iloc[i, 1:].dropna()

            for cond in conditions:
                context.append(f"{question_dict[cond].format(disease)}\n\n{answers[cond]}")

        batch_size = 128
        for i in tqdm(range(0, len(context), batch_size)):
            # find end of batch
            i_end = min(i + batch_size, len(context))
            # create IDs batch
            ids = [str(x) for x in range(i, i_end)]
            # create metadata batch
            metadatas = [{'text': text} for text in context[i:i_end]]
            # create embeddings
            xc = sentence_model.encode(context[i:i_end])
            # create records list for upsert
            records = zip(ids, xc, metadatas)
            # upsert to Pinecone
            index.upsert(vectors=records)

        # check number of records in the index
        index.describe_index_stats()

    def search(self, query: str = "medicines for fever"):

        sentence_model = self.sm

        if self.index is None:
            self.build_index()

        index = self.index

        # create the query vector
        xq = sentence_model.encode(query).tolist()

        # now query
        xc = index.query(xq, top_k = 3, include_metadata = True)
        
        return xc