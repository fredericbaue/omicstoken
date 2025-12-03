import logging
from datetime import datetime

from celery import Celery

import config
import search
import summarizer
import db

LOGGER = logging.getLogger(__name__)

celery_app = Celery(
    "omicstoken_worker",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
)


@celery_app.task(name="embed_run_task")
def embed_run_task(run_id: str, owner_id: str):
    try:
        from app import _embed_run  # Local import to avoid circular dependencies
        LOGGER.info("embed_run_task start for run %s (owner %s)", run_id, owner_id)
        result = _embed_run(run_id=run_id, expected_user_id=owner_id, trigger_rebuild=False)
        if result.get("peptides_embedded", 0) > 0:
            rebuild_job = queue_rebuild_index(run_id, owner_id)
            LOGGER.info(
                "Queued rebuild_index_task for run %s (task_id=%s)", run_id, getattr(rebuild_job, "id", None)
            )
            summary_job = queue_generate_summary(run_id)
            LOGGER.info(
                "Queued generate_summary_task for run %s (task_id=%s)", run_id, getattr(summary_job, "id", None)
            )
        try:
            completion_time = datetime.utcnow().isoformat() + "Z"
            con = db.get_db_connection(config.DATA_DIR)
            start_time = None
            run = db.get_run(con, run_id)
            if run and getattr(run, "meta", None):
                start_time = run.meta.get("upload_started_at")
            duration = None
            if start_time:
                try:
                    started_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    completed_dt = datetime.fromisoformat(completion_time.replace("Z", "+00:00"))
                    duration = (completed_dt - started_dt).total_seconds()
                except Exception as parse_err:
                    LOGGER.warning("Failed to parse upload_started_at for run %s: %s", run_id, parse_err)
            update_payload = {"embed_completed_at": completion_time}
            if duration is not None:
                update_payload["time_to_embeddings_sec"] = duration
            db.update_run_meta(con, run_id, update_payload)
            con.close()
        except Exception as timing_err:
            LOGGER.warning("Failed to record timing metadata for run %s: %s", run_id, timing_err)
        LOGGER.info("embed_run_task completed for run %s (owner %s)", run_id, owner_id)
        return result
    except Exception as e:
        LOGGER.exception("embed_run_task failed for run %s (owner %s): %s", run_id, owner_id, e)
        raise


@celery_app.task(name="rebuild_index_task")
def rebuild_index_task(source_run_id: str = None, owner_id: str = None):
    try:
        LOGGER.info(
            "rebuild_index_task start (source_run_id=%s, owner_id=%s)", source_run_id, owner_id
        )
        n_vectors = search.rebuild_faiss_index(None, config.DATA_DIR, source_run_id)
        LOGGER.info(
            "FAISS index rebuilt with %s vectors (source_run_id=%s)",
            n_vectors,
            source_run_id,
        )
        return {"status": "ok", "n_vectors": n_vectors}
    except Exception as e:
        LOGGER.exception("rebuild_index_task failed (source_run_id=%s): %s", source_run_id, e)
        raise


@celery_app.task(name="generate_summary_task")
def generate_summary_task(run_id: str):
    try:
        LOGGER.info("generate_summary_task start for run %s", run_id)
        summary = summarizer.generate_summary(run_id)
        LOGGER.info("generate_summary_task completed for run %s", run_id)
        return summary
    except Exception as e:
        LOGGER.exception("generate_summary_task failed for run %s: %s", run_id, e)
        raise


def queue_embed_run(run_id: str, owner_id: str):
    LOGGER.info(
        "queue_embed_run called for run %s (owner %s, broker=%s)",
        run_id,
        owner_id,
        config.CELERY_BROKER_URL,
    )
    try:
        job = celery_app.send_task("embed_run_task", args=[run_id, owner_id])
        LOGGER.info(
            "embed_run_task dispatched for run %s (owner %s, task_id=%s, broker=%s)",
            run_id,
            owner_id,
            job.id,
            config.CELERY_BROKER_URL,
        )
        return job
    except Exception as exc:
        LOGGER.error(
            "Failed to dispatch embed_run_task for run %s (owner %s, broker=%s): %s",
            run_id,
            owner_id,
            config.CELERY_BROKER_URL,
            exc,
        )
        raise RuntimeError(
            f"Failed to queue embed_run_task for run {run_id} (owner {owner_id})."
        ) from exc


def queue_rebuild_index(source_run_id: str = None, owner_id: str = None):
    LOGGER.info(
        "queue_rebuild_index called (source_run_id=%s, owner_id=%s, broker=%s)",
        source_run_id,
        owner_id,
        config.CELERY_BROKER_URL,
    )
    try:
        job = celery_app.send_task("rebuild_index_task", args=[source_run_id, owner_id])
        LOGGER.info(
            "rebuild_index_task dispatched (source_run_id=%s, owner_id=%s, task_id=%s, broker=%s)",
            source_run_id,
            owner_id,
            job.id,
            config.CELERY_BROKER_URL,
        )
        return job
    except Exception as exc:
        LOGGER.error(
            "Failed to dispatch rebuild_index_task (source_run_id=%s, owner_id=%s, broker=%s): %s",
            source_run_id,
            owner_id,
            config.CELERY_BROKER_URL,
            exc,
        )
        raise RuntimeError(
            f"Failed to queue rebuild_index_task for source_run_id={source_run_id or 'unknown'}."
        ) from exc


def queue_generate_summary(run_id: str):
    LOGGER.info(
        "queue_generate_summary called for run %s (broker=%s)", run_id, config.CELERY_BROKER_URL
    )
    try:
        job = celery_app.send_task("generate_summary_task", args=[run_id])
        LOGGER.info(
            "generate_summary_task dispatched for run %s (task_id=%s, broker=%s)",
            run_id,
            job.id,
            config.CELERY_BROKER_URL,
        )
        return job
    except Exception as exc:
        LOGGER.error(
            "Failed to dispatch generate_summary_task for run %s (broker=%s): %s",
            run_id,
            config.CELERY_BROKER_URL,
            exc,
        )
        raise RuntimeError(f"Failed to queue generate_summary_task for run {run_id}.") from exc
