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
        prompt = f"""
        ```markdown
        You are a SQL expert tasked with writing queries including a time attribute for the relevant table. The user wants to execute a query for the following repos: {self.repos} to answer the query of {self.query}. 

        - If you need to filter by repository, use the `repo_name` column.
        - When partitioning items by a specific time period, always use the `time_bucket` function provided by TimescaleDB. For example:
        ```sql
        SELECT time_bucket('2 month', start_ts) AS month,
                COUNT(*) AS issue_count
        FROM github_issues
        GROUP BY month
        ORDER BY month;
        ```
        - This groups data into a 2 month buckets and the individual rows into groups of two week intervals. Adjust the interval (e.g., '2 weeks', '1 day') as needed for the specific query.
        - The `time_bucket` function can take any arbitrary interval such as week, month, or year.
        - When looking at comments, note that `order_in_issue` begins with 1 and increments thereafter, so make sure to account for that.
        - The `metadata` field is currently empty, so do not use it.
        - To determine if an issue is closed or not, use the `issue_label` column.
        - To detect involvement or participation in an issue, check for comments in the `github_issue_comments` table.
        - Only use the tables and the fields provided in the database schema below.
        
        **Database Schema:**

        ```sql
        CREATE TABLE IF NOT EXISTS github_issues (
        issue_id INTEGER,
        metadata JSONB,
        text TEXT,
        repo_name TEXT,
        start_ts TIMESTAMPTZ NOT NULL,
        end_ts TIMESTAMPTZ,
        embedding VECTOR(1536) NOT NULL
        );

        CREATE INDEX github_issue_embedding_idx
        ON github_issues
        USING diskann (embedding);

        -- Create a Hypertable that breaks it down by 1 month intervals
        SELECT create_hypertable('github_issues', 'start_ts', chunk_time_interval => INTERVAL '1 month');

        CREATE TABLE github_issue_summaries (
            issue_id INTEGER,
            text TEXT,
            label issue_label NOT NULL,
            repo_name TEXT,
            embedding VECTOR(1536) NOT NULL
        );

        CREATE INDEX github_issue_summaries_embedding_idx
        ON github_issue_summaries
        USING diskann (embedding);
        ```

        Examples:
        Query: How many issues were created in the apache/spark repository every two months?        
        SQL:
        tsdb=> SELECT time_bucket('2 months', start_ts) AS two_month_period, COUNT(*) AS issue_count 
                FROM github_issues 
                WHERE repo_name = 'rust-lang/rust' 
                GROUP BY two_month_period 
                ORDER BY two_month_period;

                
            two_month_period    | issue_count 
        ------------------------+-------------
        2013-09-01 00:00:00+00 |           1
        2015-01-01 00:00:00+00 |           3
        2015-03-01 00:00:00+00 |           2
        2015-05-01 00:00:00+00 |           1

        Example:
        Query: What is the average time to first response for issues in the MicrosoftDocs/azure-docs repository?
        SQL:
        ```sql
        SELECT AVG(EXTRACT(EPOCH FROM (first_comment.created_at - issues.start_ts))) / 3600 AS average_time_to_first_response_hours
        FROM github_issues AS issues
        LEFT JOIN (
            SELECT issue_id, MIN(created_at) AS created_at
            FROM github_issue_comments
            WHERE order_in_issue = 1
            GROUP BY issue_id
        ) AS first_comment ON issues.issue_id = first_comment.issue_id
        WHERE issues.repo_name = 'MicrosoftDocs/azure-docs'
        AND first_comment.created_at IS NOT NULL;


        average_time_to_first_response_hours
        -------------------------------------
                                       12.5

        Example:
        Query: How many unique issues has alextp commented on in the tensorflow library?
        SQL:
        ```sql
        SELECT COUNT(DISTINCT issue_id) AS unique_issues_count
        FROM github_issue_comments
        WHERE author = 'alextp'
        AND repo_name = 'tensorflow/tensorflow';
        ```

        unique_issues_count
        --------------------
                           4
        """
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
                "content": "You're a helpful assistant that summarizes information about issues from a github repository. Be sure to output your response in a single paragraph that is concise and to the point.",
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
                "content": "You are an AI assistant that helps users query and analyze GitHub issues stored in a PostgreSQL database. Search for summaries when the user wants to understand the high level trends or patterns within a project. Otherwise just get the issues and return them. Only resort to SQL queries if the other tools are not able to answer the user's query.",
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
