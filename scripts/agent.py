from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional
from openai import OpenAI
from jinja2 import Template
from asyncpg import Connection
from fuzzywuzzy import process
import os
from asyncpg import Record
import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector
from typing import Iterable, Union
from asyncio import run
import instructor


def find_closest_repo(query: str, repos: list[str]) -> str | None:
    if not query:
        return None

    best_match = process.extractOne(query, repos)
    return best_match[0] if best_match[1] >= 80 else None


class SearchIssues(BaseModel):
    """
    Use this when the user wants to get original issue information from the database
    """

    query: Optional[str]
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )

    @field_validator("repo")
    def validate_repo(cls, v: str, info: ValidationInfo):
        matched_repo = find_closest_repo(v, info.context["repos"])
        if matched_repo is None:
            raise ValueError(
                f"Unable to match repo {v} to a list of known repos of {info.context['repos']}"
            )
        return matched_repo

    async def execute(self, conn: Connection, limit: int):
        if self.query:
            embedding = (
                OpenAI()
                .embeddings.create(input=self.query, model="text-embedding-3-small")
                .data[0]
                .embedding
            )
            args = [self.repo, limit, embedding]
        else:
            args = [self.repo, limit]
            embedding = None

        sql_query = Template(
            """
            SELECT *
            FROM {{ table_name }}
            WHERE repo_name = $1
            {%- if embedding is not none %}
            ORDER BY embedding <=> $3
            {%- endif %}
            LIMIT $2
            """
        ).render(table_name="github_issues", embedding=embedding)

        return await conn.fetch(sql_query, *args)


class RunSQLReturnPandas(BaseModel):
    """
    Use this function when the user wants to do time series analysis or data analysis and we don't have a tool that can supply the necessary information
    """

    query: str = Field(description="Description of user's query")
    repos: list[str] = Field(
        description="the repos to run the query on, should be in the format of 'owner/repo'"
    )

    async def execute(self, conn: Connection, limit: int):
        pass


class SearchSummaries(BaseModel):
    """
    This function retrieves summarized information about GitHub issues that match/are similar to a specific query, It's particularly useful for obtaining a quick snapshot of issue trends or patterns within a project.
    """

    query: Optional[str] = Field(description="Relevant user query if any")
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )

    @field_validator("repo")
    def validate_repo(cls, v: str, info: ValidationInfo):
        matched_repo = find_closest_repo(v, info.context["repos"])
        if matched_repo is None:
            raise ValueError(
                f"Unable to match repo {v} to a list of known repos of {info.context['repos']}"
            )
        return matched_repo

    async def execute(self, conn: Connection, limit: int):
        if self.query:
            embedding = (
                OpenAI()
                .embeddings.create(input=self.query, model="text-embedding-3-small")
                .data[0]
                .embedding
            )
            args = [self.repo, limit, embedding]
        else:
            args = [self.repo, limit]
            embedding = None

        sql_query = Template(
            """
            SELECT *
            FROM {{ table_name }}
            WHERE repo_name = $1
            {%- if embedding is not none %}
            ORDER BY embedding <=> $3
            {%- endif %}
            LIMIT $2
            """
        ).render(table_name="github_issue_summaries", embedding=embedding)

        return await conn.fetch(sql_query, *args)


class Summary(BaseModel):
    chain_of_thought: str
    summary: str


def summarize_content(issues: list[Record], query: Optional[str]):
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You're a helpful assistant that summarizes information about issues from a github repository. Be sure to output your response in a single paragraph that is concise and to the point.""",
            },
            {
                "role": "user",
                "content": Template(
                    """
                    Here are the relevant issues:
                    {% for issue in issues %}
                    - {{ issue['text'] }}
                    {% endfor %}
                    {% if query %}
                    My specific query is: {{ query }}
                    {% else %}
                    Please provide a broad summary and key insights from the issues above.
                    {% endif %}
                    """
                ).render(issues=issues, query=query),
            },
        ],
        response_model=Summary,
        model="gpt-4o-mini",
    )


async def get_conn():
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    await register_vector(conn)
    return conn


def one_step_agent(question: str, repos: list[str]):
    client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that helps users query and analyze GitHub issues stored in a PostgreSQL database. Search for summaries when the user wants to understand the trends or patterns within a project. Otherwise just get the issues and return them. Only resort to SQL queries if the other tools are not able to answer the user's query.",
            },
            {
                "role": "user",
                "content": Template(
                    """
                    Here is the user's question: {{ question }}
                    Here is a list of repos that we have stored in our database. Choose the one that is most relevant to the user's query:
                    {% for repo in repos %}
                    - {{ repo }}
                    {% endfor %}
                    """
                ).render(question=question, repos=repos),
            },
        ],
        validation_context={"repos": repos},
        response_model=Iterable[
            Union[
                RunSQLReturnPandas,
                SearchIssues,
                SearchSummaries,
            ]
        ],
    )


async def main():
    query = "What are the main issues people face with endpoint connectivity between different pods in kubernetes?"

    repos = [
        "rust-lang/rust",
        "kubernetes/kubernetes",
        "apache/spark",
        "golang/go",
        "tensorflow/tensorflow",
        "MicrosoftDocs/azure-docs",
        "pytorch/pytorch",
        "Microsoft/TypeScript",
        "python/cpython",
        "facebook/react",
        "django/django",
        "rails/rails",
        "bitcoin/bitcoin",
        "nodejs/node",
        "ocaml/opam-repository",
        "apache/airflow",
        "scipy/scipy",
        "vercel/next.js",
    ]

    resp = one_step_agent(query, repos)

    conn = await get_conn()
    limit = 10

    tools = [tool for tool in resp]
    print(tools)

    result = await tools[0].execute(conn, limit)

    summary = summarize_content(result, query)
    print(summary.summary)


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env", override=True)
    run(main())
