from datasets import load_dataset
from datetime import datetime
from openai import AsyncOpenAI
from asyncio import run, Semaphore
from tqdm.asyncio import tqdm_asyncio as asyncio
from textwrap import dedent
from instructor import from_openai
from jinja2 import Template
import os
from pgvector.asyncpg import register_vector
import asyncpg
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal, Any, Optional


class ClassifiedSummary(BaseModel):
    chain_of_thought: str
    label: Literal["OPEN", "CLOSED"]
    summary: str


class ProcessedIssue(BaseModel):
    issue_id: int
    text: str
    label: Literal["OPEN", "CLOSED"]
    repo_name: str
    embedding: Optional[list[float]]


class GithubIssue(BaseModel):
    issue_id: int
    metadata: dict[str, Any]
    text: str
    repo_name: str
    start_ts: datetime
    end_ts: Optional[datetime]
    embedding: Optional[list[float]]


def get_issues(n: int, repos: list[str]):
    dataset = (
        load_dataset("bigcode/the-stack-github-issues", split="train", streaming=True)
        .filter(lambda x: x["repo"] in repos)
        .take(n)
    )

    for row in dataset:
        start_time = None
        end_time = None
        for event in row["events"]:
            event_type = event["action"]
            timestamp = event["datetime"]
            timestamp = timestamp.replace("Z", "+00:00")

            if event_type == "opened":
                start_time = datetime.fromisoformat(timestamp)

            elif event_type == "closed":
                end_time = datetime.fromisoformat(timestamp)

            # Small Fall Back here - Some issues have no Creation event
            elif event_type == "created" and not start_time:
                start_time = datetime.fromisoformat(timestamp)

            elif event_type == "reopened" and not start_time:
                start_time = datetime.fromisoformat(timestamp)

        yield GithubIssue(
            issue_id=row["issue_id"],
            metadata={},
            text=row["content"],
            repo_name=row["repo"],
            start_ts=start_time,
            end_ts=end_time,
            embedding=None,
        )


async def batch_classify_issue(
    batch: list[GithubIssue], max_concurrent_requests: int = 20
) -> list[ProcessedIssue]:
    async def classify_issue(issue: GithubIssue, semaphore: Semaphore):
        client = from_openai(AsyncOpenAI())
        async with semaphore:
            classification = await client.chat.completions.create(
                response_model=ClassifiedSummary,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that classifies and summarizees GitHub issues. When summarizing the issues, make sure to expand on specific accronyms and add additional explanation where necessary.",
                    },
                    {
                        "role": "user",
                        "content": Template(
                            dedent(
                                """
                            Repo Name: {{ repo_name }}
                            Issue Text: {{ issue_text}}
                            """
                            )
                        ).render(repo_name=issue.repo_name, issue_text=issue.text),
                    },
                ],
                model="gpt-4o-mini",
            )
            return ProcessedIssue(
                issue_id=issue.issue_id,
                repo_name=issue.repo_name,
                text=classification.summary,
                label=classification.label,
                embedding=None,
            )

    semaphore = Semaphore(max_concurrent_requests)
    coros = [classify_issue(item, semaphore) for item in batch]
    results = await asyncio.gather(*coros)
    return results


async def batch_embeddings(
    data: list[ProcessedIssue],
    max_concurrent_calls: int = 20,
) -> list[ProcessedIssue]:
    oai = AsyncOpenAI()

    async def embed_row(
        item: ProcessedIssue,
        semaphore: Semaphore,
    ):
        async with semaphore:
            input_text = item.text if len(item.text) < 8000 else item.text[:6000]
            embedding = (
                (
                    await oai.embeddings.create(
                        input=input_text, model="text-embedding-3-small"
                    )
                )
                .data[0]
                .embedding
            )
            item.embedding = embedding
            return item

    semaphore = Semaphore(max_concurrent_calls)
    coros = [embed_row(item, semaphore) for item in data]
    results = await asyncio.gather(*coros)
    return results


async def get_conn():
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    await register_vector(conn)
    return conn


async def insert_github_issue_summaries(conn, issues: list[GithubIssue]):
    insert_query = """
    INSERT INTO github_issue_summaries (issue_id, text, label, embedding,repo_name)
    VALUES ($1, $2, $3, $4, $5)
    """
    summarized_issues = await batch_classify_issue(issues)
    embedded_summaries = await batch_embeddings(summarized_issues)

    await conn.executemany(
        insert_query,
        [
            (item.issue_id, item.text, item.label, item.embedding, item.repo_name)
            for item in embedded_summaries
        ],
    )

    print("GitHub issue summaries inserted successfully.")


async def insert_github_issues(conn, issues: list[GithubIssue]):
    insert_query = """
    INSERT INTO github_issues (issue_id, metadata, text, repo_name, start_ts, end_ts, embedding)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    """
    embedded_issues = await batch_embeddings(issues)

    await conn.executemany(
        insert_query,
        [
            (
                item.issue_id,
                json.dumps(item.metadata),
                item.text,
                item.repo_name,
                item.start_ts,
                item.end_ts,
                item.embedding,
            )
            for item in embedded_issues
        ],
    )
    print("GitHub issues inserted successfully.")


async def setup_db(conn: asyncpg.Connection):
    init_sql = """
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

    DROP TABLE IF EXISTS github_issue_summaries CASCADE;
    DROP TABLE IF EXISTS github_issues CASCADE;

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

    CREATE UNIQUE INDEX ON github_issues (issue_id, start_ts);

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
    """

    await conn.execute(init_sql)


async def process_issues(n_issues: int, repos: list[str], conn: asyncpg.Connection):
    issues = list(get_issues(n_issues, repos))
    await insert_github_issues(conn, issues)
    await insert_github_issue_summaries(conn, issues)


async def main():
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
    conn = await get_conn()
    try:
        n_issues = 400
        await setup_db(conn)
        await process_issues(n_issues, repos, conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env", override=True)
    run(main())
