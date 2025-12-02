import logging

from celery import Celery

import config
import search
import summarizer

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
        result = _embed_run(run_id=run_id, expected_user_id=owner_id)
        if result.get("peptides_embedded", 0) > 0:
            rebuild_index_task.delay()
            generate_summary_task.delay(run_id)
        return result
    except Exception as e:
        LOGGER.exception("embed_run_task failed for run %s: %s", run_id, e)
        raise


@celery_app.task(name="rebuild_index_task")
def rebuild_index_task():
    try:
        n_vectors = search.rebuild_faiss_index(None, config.DATA_DIR)
        LOGGER.info("FAISS index rebuilt with %s vectors", n_vectors)
        return {"status": "ok", "n_vectors": n_vectors}
    except Exception as e:
        LOGGER.exception("rebuild_index_task failed: %s", e)
        raise


@celery_app.task(name="generate_summary_task")
def generate_summary_task(run_id: str):
    try:
        summary = summarizer.generate_summary(run_id)
        return summary
    except Exception as e:
        LOGGER.exception("generate_summary_task failed for run %s: %s", run_id, e)
        raise
