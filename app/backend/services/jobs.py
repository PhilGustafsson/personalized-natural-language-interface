import threading
import time
import random
import requests

# Services for managing background fine-tuning jobs.

class FineTuningJob:
    def __init__(self, job_id: str, status="pending"):
        self.job_id = job_id
        self.status = status
        self.progress = 0
        self.time_remaining = -1  # in seconds, -1 means unknown
        self.result = None
    
    def start(self):
        def run_job():
            self.status = "in_progress"
            for _ in range(100):
                self.progress += 1
                self.time_remaining = (100 - self.progress) * 0.2  # Dummy estimate
                time.sleep(max(random.random() * 0.4 - 0.2, 0.02))  # Simulate work
            self.status = "done"

        threading.Thread(target=run_job).start()
    
    def get_progress(self):
        return self.progress, self.time_remaining



class FineTuningJobManager:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job: FineTuningJob):
        self.jobs[job.job_id] = job

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def update_job_status(self, job_id, status):
        job = self.get_job(job_id)
        if job:
            job.status = status
            return True
        return False

    def get_all_jobs(self):
        return list(self.jobs.values())

class STTFineTuningJob(FineTuningJob):
    def __init__(self, job_id: str, status="pending", stt_job_id: str = None):
        super().__init__(job_id, status=status)
        self.stt_job_id = stt_job_id

    def start(self):
        def run_job():
            self.status = "in_progress"

            # Wait a short moment for the STT job to appear
            time.sleep(1)

            start_time = time.time()

            while True:
                try:
                    if not self.stt_job_id:
                        # Fall back to the simple training_status endpoint if no stt_job_id
                        resp = requests.get("http://stt-app:5080/training_status", timeout=5)
                        if resp.status_code == 200:
                            data = resp.json()
                            is_training = data.get("is_training")
                            if not is_training:
                                self.status = "done"
                                self.progress = 100
                                break
                            else:
                                elapsed = time.time() - start_time
                                estimated_total = 300
                                prog = min(int((elapsed / estimated_total) * 100), 99)
                                self.progress = prog
                                self.time_remaining = max(estimated_total - elapsed, 0)
                        else:
                            print(f"Error polling STT status: {resp.status_code}")
                    else:
                        # Poll STT job endpoint to get precise status
                        url = f"http://stt-app:5080/jobs/{self.stt_job_id}"
                        resp = requests.get(url, timeout=5)
                        if resp.status_code == 200:
                            data = resp.json()
                            stt_status = data.get("status")
                            # Map STT status to our status and compute progress heuristically
                            if stt_status == "RUNNING" or stt_status == "running":
                                # Try to compute progress based on created_at timestamp if present
                                created_at = data.get("created_at")
                                elapsed = 0
                                try:
                                    if created_at:
                                        # created_at is ISO format
                                        from datetime import datetime
                                        created_ts = datetime.fromisoformat(created_at).timestamp()
                                        elapsed = time.time() - created_ts
                                except Exception:
                                    elapsed = time.time() - start_time

                                stt_progress = data.get("progress")
                                if stt_progress is not None:
                                    self.progress = stt_progress
                                    if self.progress > 0:
                                        estimated_total = (elapsed * 100) / self.progress
                                        self.time_remaining = max(estimated_total - elapsed, 0)
                                    else:
                                        self.time_remaining = 300
                                else:
                                    estimated_total = 300
                                    prog = min(int((elapsed / estimated_total) * 100), 99)
                                    self.progress = prog
                                    self.time_remaining = max(estimated_total - elapsed, 0)
                                
                                self.status = "in_progress"
                            elif stt_status == "COMPLETED" or stt_status == "completed":
                                self.status = "done"
                                self.progress = 100
                                self.result = data.get("result")
                                break
                            elif stt_status == "FAILED" or stt_status == "failed":
                                self.status = "failed"
                                self.progress = max(self.progress, 0)
                                break
                            else:
                                # PENDING or unknown
                                self.status = "in_progress"
                                self.progress = max(self.progress, 0)
                        elif resp.status_code == 404:
                            # STT job not found yet; keep waiting
                            pass
                        else:
                            print(f"Error polling STT job endpoint: {resp.status_code}")
                except Exception as e:
                    print(f"Exception polling STT status: {e}")
                    pass

                time.sleep(2) # Poll every 2 seconds

        threading.Thread(target=run_job).start()


class TTSFineTuningJob(FineTuningJob):
    """TTS fine-tuning job that polls TTS service, mirroring STT pattern."""
    
    def __init__(self, job_id: str, status="pending", tts_job_id: str = None):
        super().__init__(job_id, status=status)
        self.tts_job_id = tts_job_id

    def start(self):
        def run_job():
            self.status = "in_progress"
            time.sleep(1)  # Wait for TTS job to appear
            start_time = time.time()

            while True:
                try:
                    if not self.tts_job_id:
                        # No job ID, use simple heuristic
                        elapsed = time.time() - start_time
                        estimated_total = 600  # TTS typically takes longer than STT
                        prog = min(int((elapsed / estimated_total) * 100), 99)
                        self.progress = prog
                        self.time_remaining = max(estimated_total - elapsed, 0)
                        
                        if elapsed > estimated_total:
                            self.status = "done"
                            self.progress = 100
                            break
                    else:
                        # Poll TTS job endpoint
                        url = f"http://tts-app:8000/job_status/{self.tts_job_id}"
                        resp = requests.get(url, timeout=5)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            tts_status = data.get("status")
                            
                            if tts_status in ["RUNNING", "running"]:
                                tts_progress = data.get("progress", 0)
                                self.progress = tts_progress
                                
                                created_at = data.get("created_at")
                                elapsed = 0
                                try:
                                    if created_at:
                                        from datetime import datetime
                                        created_ts = datetime.fromisoformat(created_at).timestamp()
                                        elapsed = time.time() - created_ts
                                except Exception:
                                    elapsed = time.time() - start_time
                                
                                if self.progress > 0:
                                    estimated_total = (elapsed * 100) / self.progress
                                    self.time_remaining = max(estimated_total - elapsed, 0)
                                else:
                                    self.time_remaining = 600
                                
                                self.status = "in_progress"
                            elif tts_status in ["COMPLETED", "completed"]:
                                self.status = "done"
                                self.progress = 100
                                self.result = data.get("result")
                                break
                            elif tts_status in ["FAILED", "failed"]:
                                self.status = "failed"
                                self.progress = max(self.progress, 0)
                                break
                            else:
                                # PENDING or unknown
                                self.status = "in_progress"
                                self.progress = max(self.progress, 0)
                        elif resp.status_code == 404:
                            # TTS job not found yet; keep waiting
                            pass
                        else:
                            print(f"Error polling TTS job endpoint: {resp.status_code}")
                except Exception as e:
                    print(f"Exception polling TTS status: {e}")
                    pass

                time.sleep(2)  # Poll every 2 seconds

        threading.Thread(target=run_job).start()

    
# Initialize fine-tuning jobs manager.
job_manager = FineTuningJobManager()