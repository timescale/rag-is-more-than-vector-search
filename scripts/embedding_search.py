from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI
from jinja2 import Template
from asyncpg import Connection
import asyncpg
from pgvector.asyncpg import register_vector
from dotenv import load_dotenv
from asyncio import run
import os


class SearchIssues(BaseModel):
    """
    Use this when the user wants to get original issue information from the database
    """

    query: Optional[str]
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )

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


async def get_conn():
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    await register_vector(conn)
    return conn


async def main():
    query = (
        "What are the main problems people are facing with installation with Kubernetes"
    )

    conn = await get_conn()
    limit = 10
    resp = await SearchSummaries(query=query, repo="kubernetes/kubernetes").execute(
        conn, limit
    )

    for row in resp[:3]:
        print(row["text"])


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env", override=True)
    run(main())
