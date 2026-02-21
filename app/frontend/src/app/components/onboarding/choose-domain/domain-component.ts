import { Component, ChangeDetectorRef, OnInit, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute } from '@angular/router';
import { environment } from '../../../../environments/environment';
import { ProfileService } from '../../../services/profile.service';
import { AudioPlayerComponent } from '../../shared/audio-player/audio-player.component';
import { SecondsToMmssPipe } from '../../../pipes/seconds-to-mmss.pipe';
import { RecordingService } from '../../../services/recording.service';
import { toSignal } from '@angular/core/rxjs-interop';

@Component({
  selector: 'app-domain',
  standalone: true,
  imports: [CommonModule, AudioPlayerComponent, SecondsToMmssPipe],
  templateUrl: './domain-component.html'
})
export class DomainComponent implements OnInit {
  @Output() cancel = new EventEmitter<void>();
  domainID: string = '1';
  domainText = '';

  // Recording state (uses RecordingService)
  recording = false;
  paused = false;
  status = '';
  recordingURL: string | null = null;
  showPreview = false;
  private recordingBlob: Blob | null = null;
  elapsedTimeSignal: any;

  recordingPreviewUrl: string = '';

  constructor(private http: HttpClient, private route: ActivatedRoute, private cd: ChangeDetectorRef, private profileService: ProfileService, private recordingService: RecordingService) {
    this.elapsedTimeSignal = toSignal(this.recordingService.getElapsedTime(), {initialValue: 0});
  }

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      if (params['domainID']) this.domainID = params['domainID'];
      this.loadDomain();
    });
  }

  async loadDomain() {
    try {
      const url = `${environment.apiUrl}/audio/domains/domain${this.domainID}/`;
      const res: any = await this.http.get(url).toPromise();
      this.domainText = res.domain_text || JSON.stringify(res);
    } catch (err: any) {
      console.error('Failed to load domain:', err);
      this.domainText = 'Failed to load domain text.';
    }
  }

  // The mic should be activated during onboarding. Start recording will handle calling
  // getUserMedia if needed.

  startRecording() {
    this.recordingService.startRecording().subscribe({
      next: (state: 'idle' | 'recording' | 'stopped' | 'paused') => {
        this.recording = state === 'recording';
        this.paused = state === 'paused';
        this.status = state === 'recording' ? 'Recording...' : this.status;
      },
      error: (err: any) => {
        console.error('Recording error', err);
        this.status = 'Could not start recording: ' + (err?.message ?? err);
      }
    });
    this.showPreview = false;
    this.cd.detectChanges();
  }

  pauseOrResume() {
    if (this.paused) {
      this.recordingService.resumeRecording();
      this.paused = false;
      this.status = 'Recording...';
    } else {
      this.recordingService.pauseRecording();
      this.paused = true;
      this.status = 'Paused';
    }
  }

  finishRecording() {
    this.recordingService.stopRecording().subscribe({
      next: (blob: Blob) => {
        this.recordingBlob = blob;
        this.recordingPreviewUrl = URL.createObjectURL(blob);
        this.showPreview = true;
        this.recording = false;
        this.status = 'Recording finished. Preview available.';
        this.cd.detectChanges();
      },
      error: (err: any) => {
        console.error('Stop recording error', err);
        this.status = 'Failed to finish recording';
      }
    });
  }

  discardRecording() {
    if (this.recordingPreviewUrl) {
      URL.revokeObjectURL(this.recordingPreviewUrl);
      this.recordingPreviewUrl = '';
    }
    this.showPreview = false;
    this.recordingBlob = null;
    this.status = 'Recording discarded.';
  }

  async saveRecording() {
    if (!this.recordingBlob) { this.status = 'No recording to save'; return; }
    const blob = this.recordingBlob;
    try {
      const profileID = this.profileService.profileID();
      if (!profileID) { this.status = 'No profile selected — cannot save'; return; }
      const fd = new FormData();
      // Do not prompt for a filename; let the backend produce domain{n}_YYYYMMDD_HHMM.wav
      fd.append('file', blob, `domain${this.domainID}.wav`);
      fd.append('profileID', profileID);
      fd.append('base_name', `domain${this.domainID}`);
      fd.append('domainID', `${this.domainID}`);
      this.status = 'Uploading...';
      const url = `${environment.apiUrl}/audio/save-domain-recording/`;
      const res: any = await this.http.post(url, fd).toPromise();
      const savedFilename = res && res.filename ? res.filename : `domain${this.domainID}.wav`;
      this.status = `Saved as audio_domain/${savedFilename}`;
    } catch (err: any) {
      console.error('Upload error', err);
      this.status = 'Upload failed: ' + (err?.message ?? err);
    }
  }

  onBack() {
    this.cancel.emit();
  }

  // Cleanup not required; RecordingService handles stream lifecycle.
}
