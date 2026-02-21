import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

/**
 * Monitors the microphone input level.
 */
@Injectable({
    providedIn: 'root'
})
export class MicLevelService implements OnDestroy {
    private micLevel$ = new BehaviorSubject<number>(0); // 0–1
    private audioContext?: AudioContext;
    private analyser?: AnalyserNode;
    private dataArray?: Uint8Array;
    private meterIntervalId?: any;

    constructor() { }

    /**
     * Start monitoring the microphone.
     * Returns an observable of normalized volume levels (0–1).
     */
    async start(): Promise<Observable<number>> {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('MediaDevices API not supported.');
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        this.audioContext = new AudioContext();
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        source.connect(this.analyser);
        this.analyser.fftSize = 256;
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);

        this.meterIntervalId = setInterval(() => this.updateLevel(), 50); // 20x/sec

        return this.micLevel$.asObservable();
    }

    /**
     * Returns the current microphone level as an observable (0–1).
     */
    getMicLevel(): Observable<number> {
        return this.micLevel$.asObservable();
    }

    private updateLevel() {
        if (!this.analyser) return;

    const buffer = new Uint8Array(this.analyser.fftSize);
    this.analyser.getByteTimeDomainData(buffer);

    let sum = 0;
    for (const val of buffer) {
        const normalized = (val - 128) / 128;
        sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / buffer.length);
    this.micLevel$.next(rms);
    }

    stop() {
        console.log('Stopping MicLevelService');
        if (this.meterIntervalId) {
            clearInterval(this.meterIntervalId);
            this.meterIntervalId = undefined;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = undefined;
        }
        this.micLevel$.next(0);
    }

    ngOnDestroy() {
        this.stop();
    }
}
