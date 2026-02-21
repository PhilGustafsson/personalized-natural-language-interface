import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';

/**
 * A service for recording audio clips.
 */
@Injectable({
    providedIn: 'root'
})
export class RecordingService {
    private mediaRecorder?: MediaRecorder;
    private audioChunks: Blob[] = [];
    private stream?: MediaStream;
    private recordingState$ = new Subject<'idle' | 'recording' | 'stopped' | 'paused'>();

    private elapsedSeconds$ = new BehaviorSubject<number>(0);
    private timerId?: any;

    constructor() { }

    /**
     * Starts recording and emits state changes.
     */
    startRecording(): Observable<'idle' | 'recording' | 'stopped' | 'paused'> {
        this.initMediaRecorder();
        return this.recordingState$.asObservable();
    }

    /**
     * Stops the recording and emits the audio Blob once.
     */
    stopRecording(): Observable<Blob> {
        return new Observable<Blob>((observer) => {
            if (!this.mediaRecorder) {
                observer.error(new Error('No recording in progress.'));
                return;
            }

            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.cleanup();
                observer.next(audioBlob);
                observer.complete();
            };

            this.mediaRecorder.stop();
            this.recordingState$.next('stopped');
        });
    }

    pauseRecording() {
        if (!this.mediaRecorder || this.mediaRecorder.state !== 'recording') return;
        this.mediaRecorder.pause();
        if (this.timerId) {
            clearInterval(this.timerId);
            this.timerId = undefined;
        }
        this.recordingState$.next('paused');
    }

    resumeRecording() {
        if (!this.mediaRecorder || this.mediaRecorder.state !== 'paused') return;
        this.mediaRecorder.resume();
        this.recordingState$.next('recording');
        // resume timer
        this.timerId = setInterval(() => {
            this.elapsedSeconds$.next(this.elapsedSeconds$.value + 1);
        }, 1000);
    }

    /**
     * Utility: get a playback URL for an audio Blob.
     */
    createAudioURL(blob: Blob): string {
        console.log('Creating audio URL for blob');
        return URL.createObjectURL(blob);
    }

    /**
     * The elapsed recording time as an observable.
     */
    getElapsedTime(): Observable<number> {
        return this.elapsedSeconds$.asObservable();
    }

    private async initMediaRecorder(): Promise<void> {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('MediaDevices API not supported in this browser.');
        }

        this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        this.mediaRecorder = new MediaRecorder(this.stream);
        this.audioChunks = [];

        this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };

        this.mediaRecorder.start();
        this.recordingState$.next('recording');

        // Reset and start the timer
        this.elapsedSeconds$.next(0);
        this.timerId = setInterval(() => {
            this.elapsedSeconds$.next(this.elapsedSeconds$.value + 1);
        }, 1000);
    }

    private cleanup(): void {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        this.stream = undefined;
        this.mediaRecorder = undefined;
        this.audioChunks = [];

        if (this.timerId) {
            clearInterval(this.timerId);
            this.timerId = undefined;
        }

        this.recordingState$.next('idle');
    }
}
