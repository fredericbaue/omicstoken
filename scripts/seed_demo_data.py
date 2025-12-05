import argparse
import asyncio
import os
from datetime import datetime

import config
import auth
import db
import app
import models

DEMO_RUNS = [
    {
        "run_id": "DEMO_RUN_1",
        "name": "Demo Run 1",
        "features": [
            {"feature_id": "d1_f1", "sequence": "PEPTIDE", "intensity": 1000, "length": 7, "charge": 1, "hydrophobicity": 0.2},
            {"feature_id": "d1_f2", "sequence": "KLVFFAE", "intensity": 800, "length": 7, "charge": 1, "hydrophobicity": 0.5},
            {"feature_id": "d1_f3", "sequence": "VVVVVVV", "intensity": 1200, "length": 7, "charge": 0, "hydrophobicity": 1.0},
            {"feature_id": "d1_f4", "sequence": "RKRKRK", "intensity": 600, "length": 6, "charge": 4, "hydrophobicity": -0.2},
        ],
    },
    {
        "run_id": "DEMO_RUN_2",
        "name": "Demo Run 2",
        "features": [
            {"feature_id": "d2_f1", "sequence": "AAAAAAA", "intensity": 500, "length": 7, "charge": 0, "hydrophobicity": 0.1},
            {"feature_id": "d2_f2", "sequence": "DDDDDDD", "intensity": 900, "length": 7, "charge": -3, "hydrophobicity": -0.8},
            {"feature_id": "d2_f3", "sequence": "MKWVTFI", "intensity": 1500, "length": 7, "charge": 1, "hydrophobicity": 0.7},
            {"feature_id": "d2_f4", "sequence": "GILFVG", "intensity": 700, "length": 6, "charge": 0, "hydrophobicity": 0.6},
        ],
    },
    {
        "run_id": "DEMO_RUN_3",
        "name": "Demo Run 3",
        "features": [
            {"feature_id": "d3_f1", "sequence": "PEPPEP", "intensity": 1100, "length": 6, "charge": 1, "hydrophobicity": 0.3},
            {"feature_id": "d3_f2", "sequence": "FLIMVV", "intensity": 950, "length": 6, "charge": 0, "hydrophobicity": 0.8},
            {"feature_id": "d3_f3", "sequence": "RRAAAA", "intensity": 400, "length": 6, "charge": 2, "hydrophobicity": 0.0},
            {"feature_id": "d3_f4", "sequence": "DDGDDD", "intensity": 650, "length": 6, "charge": -2, "hydrophobicity": -0.7},
        ],
    },
]

DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo-password")


async def ensure_demo_user():
    async with auth.async_session_maker() as session:
        result = await session.execute(auth.select(auth.User).where(auth.User.email == config.DEMO_USER_EMAIL))
        user = result.scalars().first()
        if user:
            return user
        user = auth.User(
            email=config.DEMO_USER_EMAIL,
            hashed_password=auth.pwd_context.hash(DEMO_PASSWORD),
            is_active=True,
            is_verified=True,
            is_superuser=False,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


def seed_runs_for_user(user_id: str):
    con = db.get_db_connection(config.DATA_DIR)
    try:
        for run_def in DEMO_RUNS:
            run_id = run_def["run_id"]
            existing = db.get_run(con, run_id)
            if existing:
                emb = db.get_peptide_embeddings(con, run_id)
                if not emb:
                    app._embed_run(run_id, expected_user_id=user_id)
                continue

            meta = {"demo": True, "name": run_def.get("name")}
            db.insert_run(con, run_id, meta, user_id=user_id)

            for feat in run_def["features"]:
                feature_model = models.Feature(
                    feature_id=feat["feature_id"],
                    mz=0.0,
                    rt_sec=None,
                    intensity=feat.get("intensity", 0.0),
                    adduct=None,
                    polarity=None,
                    peptide_sequence=feat["sequence"],
                    annotation_score=None,
                    metadata={
                        "length": feat.get("length"),
                        "charge": feat.get("charge"),
                        "hydrophobicity": feat.get("hydrophobicity"),
                    },
                )
                db.insert_feature(con, run_id, feature_model)
            if hasattr(con, "commit"):
                con.commit()
            app._embed_run(run_id, expected_user_id=user_id)
    finally:
        con.close()


async def seed_demo_data(force: bool = False):
    if not config.DEMO_MODE and not force:
        print("DEMO_MODE is disabled. Set DEMO_MODE=true or use --force to seed demo data.")
        return
    user = await ensure_demo_user()
    seed_runs_for_user(str(user.id))
    print(f"Seeded demo data for {config.DEMO_USER_EMAIL}")


def main():
    parser = argparse.ArgumentParser(description="Seed demo data (demo user + runs).")
    parser.add_argument("--force", action="store_true", help="Seed even if DEMO_MODE is False.")
    args = parser.parse_args()
    asyncio.run(seed_demo_data(force=args.force))


if __name__ == "__main__":
    main()
