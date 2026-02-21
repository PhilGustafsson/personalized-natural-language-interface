import { Component, EventEmitter, Output, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { ProfileService } from '../../../services/profile.service';
import { AudioPlayerComponent } from '../../shared/audio-player/audio-player.component';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { TextareaModule } from 'primeng/textarea';
import { SecondsToMmssPipe } from '../../../pipes/seconds-to-mmss.pipe';

@Component({
  selector: 'app-write-own-prompt',
  standalone: true,
  imports: [CommonModule, FormsModule, AudioPlayerComponent, ButtonModule, InputTextModule, TextareaModule, SecondsToMmssPipe],
  templateUrl: './write-own-prompt.component.html'
})
export class WriteOwnPromptComponent {
  @Output() close = new EventEmitter<void>();

  // Script text
  scriptText: string = '';
  scriptName: string = '';

  // Saved scripts
  savedScripts: Array<{name: string, text: string}> = [];
  loadingScripts = false;
  showScriptSelector = false;

  // Recording state
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

  // UI state
  mode: 'write' | 'record' = 'write';
  savedScriptId: string | null = null;

  constructor(
    private http: HttpClient,
    private profileService: ProfileService,
    private cd: ChangeDetectorRef
  ) {
    this.loadSavedScripts();
  }

  goToRecord() {
    this.saveScript();
  }

  // --- Script writing ---
  async saveScript(): Promise<void> {
    if (!this.scriptText.trim()) {
      this.status = 'Script cannot be empty';
      return;
    }

    if (!this.scriptName.trim()) {
      this.status = 'Script name is required';
      return;
    }

    try {
      const profileId = this.profileService.profileID();
      if (!profileId) {
        this.status = 'No profile selected — cannot save';
        return;
      }

      const payload = {
        profileID: profileId,
        script_name: this.scriptName,
        script_text: this.scriptText
      };

      const url = `${environment.apiUrl}/audio/save-script/`;
      this.status = 'Saving script...';
      const res: any = await this.http.post(url, payload).toPromise();
      
      this.savedScriptId = res?.script_id || this.scriptName;
      this.status = `Script saved as ${this.scriptName}`;
      this.mode = 'record';
    } catch (err: any) {
      console.error('save script error', err);
      this.status = 'Failed to save script: ' + (err?.message ?? err);
    }
  }

  editScript(): void {
    this.mode = 'write';
    this.savedScriptId = null;
  }

  // --- Load saved scripts ---
  loadSavedScripts(): void {
    this.loadingScripts = true;
    const profileId = this.profileService.profileID();
    if (!profileId) {
      this.loadingScripts = false;
      return;
    }

    const url = `${environment.apiUrl}/audio/get-scripts/`;
    this.http.get<any>(url, { params: { profileID: profileId } }).subscribe({
      next: (res: any) => {
        this.savedScripts = res.scripts || [];
        this.loadingScripts = false;
      },
      error: (err: any) => {
        console.error('Failed to load scripts:', err);
        this.savedScripts = [];
        this.loadingScripts = false;
      }
    });
  }

  toggleScriptSelector(): void {
    this.showScriptSelector = !this.showScriptSelector;
  }

  loadScript(scriptName: string, scriptText: string): void {
    this.scriptName = scriptName;
    this.scriptText = scriptText;
    this.showScriptSelector = false;
  }

  // --- Microphone and recording controls ---
  async activateMic(): Promise<void> {
    if (this.micActive) {
      this.status = 'Microphone already active';
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaStream = stream;
      this.micActive = true;
      this.status = 'Microphone active';
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

  startRecording(): void {
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

    this.cd.detectChanges();

    this.timerInterval = setInterval(() => {
      this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
      this.cd.detectChanges();
    }, 500);
    this.elapsedTime = 0;
    this.cd.detectChanges();
  }

  pauseOrResume(): void {
    if (!this.mediaRecorder) {
      this.status = 'No active recorder';
      return;
    }
    if (this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.pause();
      this.paused = true;
      this.status = 'Paused';
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
    } else if (this.mediaRecorder.state === 'paused') {
      this.mediaRecorder.resume();
      this.paused = false;
      this.status = 'Recording...';
      this.timerInterval = setInterval(() => {
        this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
        this.cd.detectChanges();
      }, 500);
    } else {
      this.status = `Recorder state: ${this.mediaRecorder.state}`;
    }
  }

  async finishRecording(): Promise<void> {
    if (!this.mediaRecorder) {
      this.status = 'No active recorder to finish';
      return;
    }
    if (this.mediaRecorder.state === 'inactive') {
      this.status = 'Recorder is not active';
      return;
    }

    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }

    this.mediaRecorder.stop();

    // Wait for mediaRecorder to finish processing
    const blob = new Blob(this.chunks, { type: 'audio/webm;codecs=opus' });
    this.recordingPreviewUrl = URL.createObjectURL(blob);
    console.log('Created blob URL:', this.recordingPreviewUrl, 'Blob size:', blob.size);
    this.showPreview = true;
    this.recording = false;
    this.status = 'Recording finished. Preview the recording below.';
    this.cd.detectChanges();
  }

  discardRecording(): void {
    if (this.recordingPreviewUrl) {
      URL.revokeObjectURL(this.recordingPreviewUrl);
      this.recordingPreviewUrl = '';
    }
    this.showPreview = false;
    this.chunks = [];
    this.status = 'Recording discarded. You can record again.';
  }

  async saveRecording(): Promise<void> {
    if (this.chunks.length === 0) {
      this.status = 'No recording to save';
      return;
    }

    const blob = new Blob(this.chunks, { type: 'audio/wav' });

    const defaultName = `script-${this.savedScriptId || 'recording'}-${new Date().toISOString().replace(/[:.]/g, '-')}`;
    const filename = window.prompt('Enter filename for this recording', defaultName);
    if (!filename) {
      this.status = 'Save cancelled (no filename provided)';
      return;
    }

    try {
      const profileId = this.profileService.profileID();
      if (!profileId) {
        this.status = 'No profile selected — cannot save';
        return;
      }

      const fd = new FormData();
      fd.append('file', blob, `${filename}.wav`);
      fd.append('profileID', profileId);
      fd.append('base_name', filename);
      fd.append('script_name', this.scriptName);
      fd.append('script_text', this.scriptText);

      const url = `${environment.apiUrl}/audio/save-own-prompt-recording/`;
      this.status = 'Uploading...';
      const res: any = await this.http.post(url, fd).toPromise();
      const savedFilename = (res && res.filename) ? res.filename : `${filename}.wav`;
      this.status = `Saved as scripts/${savedFilename}`;

    } catch (err: any) {
      console.error('upload error', err);
      this.status = 'Upload failed: ' + (err?.message ?? err);
    }
  }

  cancelAndClose(): void {
    this._cleanupStream();
    this.close.emit();
  }

  goBackToWrite(): void {
    this._cleanupStream();
    this.editScript();
  }

  private _cleanupStream(): void {
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
