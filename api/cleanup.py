from datetime import datetime, timedelta
from ..storage import database, models
from .routes import trigger_queue_worker


def run_startup_cleanup():
    """
    On server startup:
    1. Only fail jobs stuck in 'extracting', 'correcting', or 'processing' longer than 30 minutes.
    2. Reset recently interrupted jobs (< 30 mins) to 'pending' and re-queue all pending ones.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=30)
    with database.SessionLocal() as db:
        # 1. Fail jobs stuck > 30 minutes
        stuck_jobs = db.query(models.Job).filter(
            models.Job.status.in_(["extracting", "correcting", "processing"]),
            models.Job.created_at < cutoff
        ).all()
        for j in stuck_jobs:
            j.status = "failed"
            j.stage = "failed"
            j.error_message = "Job timed out: processing exceeded 30 minutes before server restart"

        # 2. Re-queue recently interrupted jobs (< 30 minutes)
        interrupted_jobs = db.query(models.Job).filter(
            models.Job.status.in_(["extracting", "correcting", "processing"]),
            models.Job.created_at >= cutoff
        ).all()
        for j in interrupted_jobs:
            j.status = "pending"
            j.stage = "pending"
            j.error_message = None

        db.commit()

    # 3. Trigger worker to process all pending jobs in the queue
    trigger_queue_worker()
