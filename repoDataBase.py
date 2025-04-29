import requests
import datetime
import os
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv(dotenv_path="token.env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def searchRepositories(language="Python", stars=">1000", per_page=5):
    query = f"language:{language} stars:{stars}"
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page={per_page}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()["items"]

def fetchReadme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        readme_json = response.json()
        return requests.get(readme_json["download_url"]).text
    return ""

def fetchIssuesFirstMonth(owner, repo, created_at):
    since = created_at
    until = (datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ") + datetime.timedelta(days=30)).isoformat() + "Z"
    url = f"https://api.github.com/repos/{owner}/{repo}/issues?since={since}&until={until}&per_page=100"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    issues = response.json()
    results = []
    for issue in issues:
        if "pull_request" not in issue:
            results.append({
                "title": issue["title"],
                "body": issue.get("body", "")
            })
    return results

def buildDataset(max_repos=5):
    data = []
    repos = searchRepositories(per_page=max_repos)
    for repo in tqdm(repos):
        owner = repo["owner"]["login"]
        repo_name = repo["name"]
        created_at = repo["created_at"]
        try:
            readme = fetchReadme(owner, repo_name)
            issues = fetchIssuesFirstMonth(owner, repo_name, created_at)
            if readme and issues and len(issues) > 5 and len(readme) > 20:
                data.append({
                    "repo_name": f"{owner}/{repo_name}",
                    "created_at": created_at,
                    "readme": readme,
                    "issues": issues
                })
        except Exception as e:
            print(f"Error processing {owner}/{repo_name}: {e}")
    return data

if __name__ == "__main__":
    dataset = buildDataset(max_repos=3000)
    with open("github_readme_issues_model_ready.jsonl", "w") as f:
        for item in dataset:
            issue_text = ""
            for i, issue in enumerate(item["issues"]):
                issue_text += f"ISSUE {i+1}:\n{issue['title']}\n{issue['body']}\n\n"
            json.dump({
                "input": f"README:\n{item['readme']}",
                "output": issue_text.strip()
            }, f)
            f.write("\n")

    print("✅ Model-ready dataset saved to github_readme_issues_model_ready.jsonl")

