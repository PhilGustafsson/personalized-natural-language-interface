import { Component, EventEmitter, Output, ChangeDetectorRef, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { ProfileService } from '../../../services/profile.service';
import { AudioPlayerComponent } from '../../shared/audio-player/audio-player.component';
import { WriteOwnPromptComponent } from '../write-own-prompt/write-own-prompt.component';
import { SecondsToMmssPipe } from '../../../pipes/seconds-to-mmss.pipe';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-transcribe',
  standalone: true,
  imports: [CommonModule, AudioPlayerComponent, SecondsToMmssPipe, WriteOwnPromptComponent, ButtonModule],
  templateUrl: './transcribe-component.html'
})
export class MainMenuOnboardingComponent implements OnInit {
  @Input() startMode: 'choose' | 'transcribe' | 'write' = 'choose';
  @Output() close = new EventEmitter<void>();

  // Modes
  mode: 'choose' | 'transcribe' | 'write' = 'choose';

  // Microphone / recording state
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  public chunks: BlobPart[] = [];

  micActive = false;
  recording = false;
  paused = false;
  status = '';
  elapsedTime = 0;
  private startTime: number = 0;
  private timerInterval: any = null;

  // Playback preview state
  recordingPreviewUrl: string = '';
  showPreview = false;

  constructor(private http: HttpClient, private profileService: ProfileService, private cd: ChangeDetectorRef) {}

  ngOnInit() {
      if (this.startMode) {
          this.mode = this.startMode;
      }
  }

  // --- Mode selection ---
  selectMode(m: 'transcribe'|'write') {
    this.mode = m === 'transcribe' ? 'transcribe' : 'write';
    this.status = '';
  }

  backToModes() {
    this.mode = 'choose';
    this.status = '';
  }

  // --- Microphone and recording controls ---
  async activateMic() {
    if (this.micActive) {
      this.status = 'Microphone already active';
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaStream = stream;
      this.micActive = true;
      this.status = 'Microphone active';
      // Prepare MediaRecorder but do not start yet
      this.mediaRecorder = new MediaRecorder(stream);
      this.mediaRecorder.ondataavailable = (ev: BlobEvent) => {
        if (ev.data && ev.data.size > 0) this.chunks.push(ev.data);
      };
      this.mediaRecorder.onstop = () => {
        this.recording = false;
        this.paused = false;
      };
    } catch (err: any) {
      console.error('activateMic error', err);
      this.status = 'Could not activate microphone: ' + (err?.message ?? err);
    }
  }

  startRecording() {
    if (!this.mediaRecorder) {
      this.status = 'Activate microphone first';
      return;
    }
    if (this.mediaRecorder.state === 'recording') {
      this.status = 'Already recording';
      return;
    }
    this.chunks = [];
    this.mediaRecorder.start();
    this.recording = true;
    this.paused = false;
    this.status = 'Recording...';
    this.showPreview = false;
    this.elapsedTime = 0;
    this.startTime = Date.now();

    // Ensure UI updates immediately
    this.cd.detectChanges();

    // Update timer every 500ms for a snappier UI and update immediately
    this.timerInterval = setInterval(() => {
      this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
      // ensure change detection picks up the change promptly
      this.cd.detectChanges();
    }, 500);
    // immediate tick so user sees 0s right away
    this.elapsedTime = 0;
    this.cd.detectChanges();
  }

  pauseOrResume() {
    if (!this.mediaRecorder) {
      this.status = 'No active recorder';
      return;
    }
    if (this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.pause();
      this.paused = true;
      this.status = 'Paused';
      // Clear timer when pausing so elapsed time stops
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
    } else if (this.mediaRecorder.state === 'paused') {
      this.mediaRecorder.resume();
      this.paused = false;
      this.status = 'Recording...';
      // Resume timer when resuming
      this.timerInterval = setInterval(() => {
        this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
        this.cd.detectChanges();
      }, 500);
    } else {
      this.status = `Recorder state: ${this.mediaRecorder.state}`;
    }
  }

  async finishRecording() {
    if (!this.mediaRecorder) {
      this.status = 'No active recorder to finish';
      return;
    }
    if (this.mediaRecorder.state === 'inactive') {
      this.status = 'Recorder is not active';
      return;
    }

    // Clear the timer
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    // stop -> onstop handler will be invoked
    this.mediaRecorder.stop();

    // assemble blob - use the MIME type that matches the MediaRecorder output
    const mimeType = this.mediaRecorder.mimeType || 'audio/webm;codecs=opus';
    const blob = new Blob(this.chunks, { type: mimeType });
    
    // Create preview URL for playback
    this.recordingPreviewUrl = URL.createObjectURL(blob);
    console.log('Created blob URL:', this.recordingPreviewUrl, 'Blob size:', blob.size);
    this.showPreview = true;
    this.recording = false;
    this.status = 'Recording finished. Preview the recording below.';
    this.cd.detectChanges();
  }

  cancelAndClose() {
    this._cleanupStream();
    this.close.emit();
  }

  discardRecording() {
    // Clean up preview URL
    if (this.recordingPreviewUrl) {
      URL.revokeObjectURL(this.recordingPreviewUrl);
      this.recordingPreviewUrl = '';
    }
    this.showPreview = false;
    this.chunks = [];
    this.status = 'Recording discarded. You can record again.';
  }

  async saveRecording() {
    if (this.chunks.length === 0) {
      this.status = 'No recording to save';
      return;
    }

    // assemble blob with audio/wav MIME type
    const blob = new Blob(this.chunks, { type: 'audio/wav' });

    // prompt for filename
    const defaultName = `transcribe-${new Date().toISOString().replace(/[:.]/g,'-')}`;
    const filename = window.prompt('Enter filename for this recording', defaultName);
    if (!filename) {
      this.status = 'Save cancelled (no filename provided)';
      return;
    }

    // upload to backend under profile/audio_transcribe/filename
    try {
      const profileId = this.profileService.profileID(); // signal accessor
      if (!profileId) {
        this.status = 'No profile selected — cannot save';
        return;
      }
      const fd = new FormData();
      fd.append('file', blob, `${filename}.wav`);
      fd.append('profileID', profileId);
      fd.append('base_name', filename);

      // Use the new audio endpoint for transcribe recordings
      const url = `${environment.apiUrl}/audio/save-transcribe-recording/`;
      this.status = 'Uploading...';
      const res: any = await this.http.post(url, fd).toPromise();
      const savedFilename = (res && res.filename) ? res.filename : `${filename}.wav`;
      this.status = `Saved as audio_transcribe/${savedFilename}`;
      
      // Stay in preview mode instead of cleaning up
    } catch (err: any) {
      console.error('upload error', err);
      this.status = 'Upload failed: ' + (err?.message ?? err);
    }
  }

  private _cleanupStream() {
    try {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        try { this.mediaRecorder.stop(); } catch {}
      }
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(t => t.stop());
      }
      if (this.recordingPreviewUrl) {
        URL.revokeObjectURL(this.recordingPreviewUrl);
        this.recordingPreviewUrl = '';
      }
    } catch (e) {
      // ignore
    }
    this.mediaStream = null;
    this.mediaRecorder = null;
    this.micActive = false;
    this.recording = false;
    this.paused = false;
    this.showPreview = false;
  }
}
