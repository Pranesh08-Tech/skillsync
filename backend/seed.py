"""
seed.py
-------
Sample developer code-snippet data for seeding the SkillSyncEngine index.

SEED_DEVELOPERS is a list of dicts consumed by both:
  - app.py  (via the "Seed sample developer code" button in the UI)
  - __main__ (direct CLI seed: python -m backend.seed)
"""

from __future__ import annotations

SEED_DEVELOPERS: list[dict[str, str]] = [
    {
        "dev_name": "Priya Sharma",
        "code_snippet": (
            "async def migrate_users_db(): "
            "db.execute('ALTER TABLE users ADD COLUMN oauth_token TEXT;')"
        ),
        "repo_name": "backend-core-service",
    },
    {
        "dev_name": "Alex Rivera",
        "code_snippet": (
            "def setup_rate_limiter(app): "
            "limiter = Limiter(key_func=get_remote_address); "
            "app.state.limiter = limiter"
        ),
        "repo_name": "api-gateway",
    },
    {
        "dev_name": "Jordan Lee",
        "code_snippet": (
            "def train_classifier(X_train, y_train): "
            "model = RandomForestClassifier(n_estimators=100); "
            "model.fit(X_train, y_train); return model"
        ),
        "repo_name": "ml-pipeline",
    },
    {
        "dev_name": "Sam Patel",
        "code_snippet": (
            "const fetchUserProfile = async (userId: string) => { "
            "const res = await fetch(`/api/users/${userId}`); "
            "return res.json(); }"
        ),
        "repo_name": "frontend-portal",
    },
    {
        "dev_name": "Taylor Kim",
        "code_snippet": (
            "resource 'aws_s3_bucket' 'artifacts' { "
            "bucket = 'skillsync-artifacts'; "
            "versioning { enabled = true } }"
        ),
        "repo_name": "infra-terraform",
    },
]


def seed_database() -> None:
    """Seed the persistent ChromaDB index with SEED_DEVELOPERS snippets."""
    # Import here to avoid circular imports when used as a library
    from backend.engine import SkillSyncEngine

    engine = SkillSyncEngine()
    for entry in SEED_DEVELOPERS:
        engine.add_developer_code(
            dev_name=entry["dev_name"],
            code_snippet=entry["code_snippet"],
            repo_name=entry["repo_name"],
        )
    print(
        f"Database seeded successfully with {len(SEED_DEVELOPERS)} "
        "developer code profiles!"
    )


if __name__ == "__main__":
    seed_database()
