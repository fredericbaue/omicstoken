from dotenv import load_dotenv

# Load .env before any tests run
load_dotenv()

# Stub Celery tasks so tests do not attempt to run a broker/worker.
import worker


class _DummyTask:
    @staticmethod
    def delay(*args, **kwargs):
        return {"status": "queued"}


worker.embed_run_task = _DummyTask()
worker.rebuild_index_task = _DummyTask()
worker.generate_summary_task = _DummyTask()
