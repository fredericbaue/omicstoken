import logging
import sqlite3
from datetime import datetime

from celery import Celery

import config
import search
import summarizer
import db
import pipeline
from db import insert_protein_structure

LOGGER = logging.getLogger(__name__)

celery_app = Celery(
    "omicstoken_worker",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
)


def _run_structure_analysis(run_id: str, feature_id: str, sequence: str, engine: str = "mock"):
    """
    Lightweight placeholder that fabricates structure metadata so Celery plumbing
    can persist results while the real folding engine is wired up.
    """
    safe_sequence = sequence or ""
    pdb_lines = [
        f"HEADER    MOCK STRUCTURE RUN {run_id}",
        f"REMARK    FEATURE {feature_id}",
        f"REMARK    ENGINE {engine}",
        f"SEQRES   1 {safe_sequence}",
        "END",
    ]
    pseudo_score = round(min(100.0, 40.0 + len(safe_sequence) * 1.25), 2) if safe_sequence else 0.0
    return {
        "pdb_id": f"{run_id}:{feature_id}",
        "sequence": safe_sequence,
        "pdb_content": "\n".join(pdb_lines),
        "plddt_score": pseudo_score,
        "engine": engine,
    }


@celery_app.task(name="embed_run_task")
def embed_run_task(run_id: str, owner_id: str):
    try:
        LOGGER.info("embed_run_task start for run %s (owner %s)", run_id, owner_id)
        result = pipeline.run_embedding_pipeline(
            run_id=run_id, expected_user_id=owner_id, trigger_rebuild=False
        )
        embeddings_written = result.get("peptides_embedded", 0)
        followup_pending = embeddings_written > 0
        if followup_pending:
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
            update_payload = {
                "embed_completed_at": completion_time,
                "embedding_pending": False,
                "indexing_pending": followup_pending,
                "summary_pending": followup_pending,
            }
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
        if source_run_id:
            meta_con = None
            try:
                meta_con = db.get_db_connection(config.DATA_DIR)
                db.update_run_meta(
                    meta_con,
                    source_run_id,
                    {
                        "indexing_pending": False,
                        "index_rebuilt_at": datetime.utcnow().isoformat() + "Z",
                    },
                )
            except Exception as meta_err:
                LOGGER.warning(
                    "Failed to update indexing metadata for run %s after rebuild: %s",
                    source_run_id,
                    meta_err,
                )
            finally:
                if meta_con:
                    meta_con.close()
        LOGGER.info(
            "FAISS index rebuilt with %s vectors (source_run_id=%s)",
            n_vectors,
            source_run_id,
        )
        return {"status": "ok", "n_vectors": n_vectors}
    except Exception as e:
        LOGGER.exception("rebuild_index_task failed (source_run_id=%s): %s", source_run_id, e)
        raise


@celery_app.task(name="protein_structure_task")
def protein_structure_task(run_id: str, feature_id: str, engine: str = "mock"):
    """
    Generate/collect protein structure analysis for a peptide and persist it.
    """
    LOGGER.info(
        "protein_structure_task start for run %s feature %s (engine=%s)",
        run_id,
        feature_id,
        engine,
    )
    con = db.get_db_connection(config.DATA_DIR)
    try:
        feature_props = db.get_feature_properties(con, run_id, feature_id)
        if not feature_props or not feature_props[0]:
            LOGGER.warning(
                "No sequence available for run %s feature %s; skipping structure task",
                run_id,
                feature_id,
            )
            return {
                "status": "skipped",
                "reason": "feature_not_found",
                "run_id": run_id,
                "feature_id": feature_id,
            }

        sequence = feature_props[0]
        analysis_payload = _run_structure_analysis(run_id, feature_id, sequence, engine)
        pdb_id = analysis_payload.get("pdb_id") or f"{run_id}:{feature_id}"

        try:
            insert_protein_structure(
                con,
                run_id=run_id,
                feature_id=feature_id,
                sequence=analysis_payload["sequence"],
                pdb_content=analysis_payload["pdb_content"],
                plddt_score=analysis_payload["plddt_score"],
                engine=analysis_payload.get("engine", engine),
            )
            con.commit()
            LOGGER.info("Structure saved to database for %s", pdb_id)
        except sqlite3.Error as db_err:
            LOGGER.error("Failed to save protein structure for %s: %s", pdb_id, db_err)

        LOGGER.info(
            "protein_structure_task completed for run %s feature %s", run_id, feature_id
        )
        return {
            **analysis_payload,
            "run_id": run_id,
            "feature_id": feature_id,
            "status": "ok",
        }
    except Exception as e:
        LOGGER.exception(
            "protein_structure_task failed for run %s feature %s: %s",
            run_id,
            feature_id,
            e,
        )
        raise
    finally:
        con.close()


@celery_app.task(name="generate_summary_task")
def generate_summary_task(run_id: str):
    summary = None
    summary_error = None
    try:
        LOGGER.info("generate_summary_task start for run %s", run_id)
        summary = summarizer.generate_summary(run_id)
        LOGGER.info("generate_summary_task completed for run %s", run_id)
        return summary
    except Exception as e:
        LOGGER.exception("generate_summary_task failed for run %s: %s", run_id, e)
        summary_error = str(e)
        raise
    finally:
        meta_con = None
        try:
            meta_con = db.get_db_connection(config.DATA_DIR)
            update_payload = {
                "summary_pending": False,
            }
            if summary_error:
                update_payload["last_summary_error"] = summary_error
            elif summary and isinstance(summary, dict) and "error" in summary:
                update_payload["last_summary_error"] = summary.get("error")
            else:
                update_payload["summary_generated_at"] = datetime.utcnow().isoformat() + "Z"
                update_payload["last_summary_error"] = None
            db.update_run_meta(meta_con, run_id, update_payload)
        except Exception as meta_err:
            LOGGER.warning("Failed to update summary metadata for run %s: %s", run_id, meta_err)
        finally:
            if meta_con:
                meta_con.close()


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
