# RAG is more than vector search

# Introduction

This is a repository that contains the code for the article `RAG is more than embeddings`. Head over to the [Timescale blog](https://www.timescale.com/blog/tag/ai/) to read the article if you haven't already. The code is compatible for python >= 3.9.

## Instructions

1. First install all the required dependencies in the `requirements.txt` file

```
pip install -r requirements.txt
```

2. Make sure to create a `.env` file that has the same environment variables as our `.env.example ` file. You can get your DB_URL after creating a Timescale instance by following the instructions [here](https://docs.timescale.com/getting-started/latest/services/#create-your-timescale-account).

3. Next, ingest in some Github Issues from the `bigcode/the-stack-github-issues` dataset by running the `scripts/ingest.py` file. This will crawl the first 100 issues that match the list of whitelisted repos in our file. We can do so by running the command below.

```bash
python3 ./scripts/ingest.py
```

3. We can then test the function calling ability of our model by running the `scripts/eval.py` file to verify that our model is choosing the right tool with respect to a user query. We can do so by running the command below.

```bash
python3 ./scripts/eval.py
```

4. In order to perform embedding search, we can define a new `.execute` function inside our tools themselves. This allows us to call a `.execute()` function when the tool is selected to immediately return a list of relevant results. To see this in action, run the command below and we'll fetch the top 10 relevant summaries from our database related to the `kubernetes/kubernetes` repository using embedding search.

```bash
python3 ./scripts/embedding_search.py
```

5. Lastly, we'll put it all together in the `agent.py` file where we'll create a one-step agent that'll be able to answer questions about specific repositories in our database. We can run this agent by executing the command below.

```bash
python3 ./scripts/agent.py
```
