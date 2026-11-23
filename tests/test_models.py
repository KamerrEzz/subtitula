from whisper_studio.models import Job, JobKind, JobStatus


def test_status_labels_are_spanish() -> None:
    assert JobStatus.PENDING == "Pendiente"
    assert JobStatus.SKIPPED == "Omitido"


def test_job_elapsed() -> None:
    job = Job(kind=JobKind.LOCAL, source="x", title="t", mode="audio_transcribe")
    assert job.elapsed == 0.0
    job.started_at = 10.0
    job.finished_at = 15.5
    assert job.elapsed == 5.5
    assert len(job.id) == 32
