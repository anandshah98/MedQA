# MedQA - Assistant

***Answer medical queries through a simple LLM chatbot rather than searching the web and piecing it together on your own***

[Access on the model on Hugging Face spaces](https://huggingface.co/spaces/xpsychted/MedQA-Assistant) - `Response time is 2 mins on average`


## How to set up the environment


#### Step one: Install the required dependencies
```
pip install transformers accelerate
pip install -qU pinecone-client[grpc] sentence-transformers
pip install gradio torch pandas openpyxl tqdm 
```

OR

```
pip install -r requirements.txt
```


#### Step two: 
   - Set up secret/environment variables `PINECONE_API_KEY` to the value of the Pinecone API key, and `PINECONE_ENV` to the value of 'Environment' in Pinecone
   - In `pinecone_integration.py`, change
   ```
   PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', None)            
   PINECONE_ENV = os.environ.get('PINECONE_ENV', "us-west4-gcp")
   ```

<br>

## NOTE - How to use memory sharding with Kaggle notebooks:


#### 1. In Kaggle, select Accelerator = 'None', then turn on the notebook
 
  - #### Run `shard_model.py` or in `app.py`, run the below code

```
PI = PineconeIndex()
PI.build_index()
qamodel = QAModel()
model, tokenizer = qamodel.load_sharded_model()
```

`If RAM capacity is not enough to fit the model, use another platform with larger capacity to shard the model and then download and paste the folder in the root directory`

#### 2. Shutdown the notebook

#### 3. Then, select Accelerator = 'GPU P100', then turn on the notebook
 
- #### Run `app.py`

<br>

## Problems we faced:

- ### Reliable medical database with proper information
  - Created a custom-curated medical database that had information about common to rare diseases
    
- ### Resource constraint
  - Used Kaggle notebook with GPU, TPU, and higher-capacity CPU support
    
- ### Memory constraint
  - Smaller LLMs do not respond well to queries and prompts, and larger LLMs are difficult to fit in memory, even on Kaggle's P100 GPU with 13GB VRAM
  - With the help of the Hugging Face Accelerate library, we used memory sharding that allowed the CPU to load pieces of the model as and when required
  - This solved our issue of not using a larger LLM, and we could use Google's flan-t5-xl as the base model.

- ### Context awareness
  - Pinecone has a great library to use for vector embedding-related tasks
  - We used Pinecone to store the medical database and retrieve the relevant documents according to the query input by the user

- ### Access constraint
  - To showcase the model, we initially used FastAPI combined with ngrok to create a public access API for the website and used Gradio on the user side to access the chatbot
  - We then used only the Gradio interface with `.launch(share=True)` embedded in the server-side program itself to create a public interface which removed the usage of FastAPI and ngrok
