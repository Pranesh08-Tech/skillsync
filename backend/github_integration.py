"""
github_integration.py
---------------------
Fetches contributor and repository data from the GitHub API.
Provides helpers for extracting commit authors, languages used,
and file-level contributions to build the skill graph.
"""

import os
import requests
from typing import Any

GITHUB_API_BASE = "https://api.github.com"


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_repo_contributors(owner: str, repo: str) -> list[dict[str, Any]]:
    """Return a list of contributor objects for the given repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contributors"
    response = requests.get(url, headers=_headers(), timeout=15)
    response.raise_for_status()
    return response.json()


def get_repo_languages(owner: str, repo: str) -> dict[str, int]:
    """Return a mapping of language -> bytes of code for the repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/languages"
    response = requests.get(url, headers=_headers(), timeout=15)
    response.raise_for_status()
    return response.json()


def get_contributor_commits(
    owner: str, repo: str, author: str, per_page: int = 30
) -> list[dict[str, Any]]:
    """Return recent commits by a specific author in the repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    params = {"author": author, "per_page": per_page}
    response = requests.get(url, headers=_headers(), params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def get_repo_tree(owner: str, repo: str, branch: str = "main") -> list[dict[str, Any]]:
    """Return the full file tree for a repository branch (flat list)."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}"
    params = {"recursive": "1"}
    response = requests.get(url, headers=_headers(), params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data.get("tree", [])


def build_contributor_skill_summary(
    owner: str, repo: str
) -> list[dict[str, Any]]:
    """
    Combine contributor list with repo languages to produce a lightweight
    skill-summary record per contributor.

    Returns a list of dicts with keys:
        login, contributions, repo_languages
    """
    contributors = get_repo_contributors(owner, repo)
    languages = get_repo_languages(owner, repo)
    lang_list = list(languages.keys())

    summaries = []
    for c in contributors:
        summaries.append(
            {
                "login": c.get("login", ""),
                "contributions": c.get("contributions", 0),
                "repo_languages": lang_list,
                "avatar_url": c.get("avatar_url", ""),
                "html_url": c.get("html_url", ""),
            }
        )
    return summaries
