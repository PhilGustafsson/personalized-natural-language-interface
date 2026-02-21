import { NgTemplateOutlet, CommonModule } from '@angular/common';
import { Component, effect, ElementRef, EventEmitter, input, Input, OnDestroy, OnInit, Output, ViewChild } from '@angular/core';
import WaveSurfer from 'wavesurfer.js';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-audio-player',
  standalone: true,
  imports: [CommonModule, ButtonModule],
  templateUrl: './audio-player.component.html'
})
export class AudioPlayerComponent implements OnInit, OnDestroy {
    readonly src = input('');
    @Input() buttonBottom = false;
    @ViewChild('waveformContainer', { static: true }) waveformRef!: ElementRef;

    @Output() playEvent = new EventEmitter<void>(); // Emits when user presses play

    waveSurfer!: WaveSurfer;
    isPlaying = false;
    currentTime = 0; // seconds

    private intervalId: any;

    constructor() {
        effect(() => {
            const url = this.src();
            console.log('Audio player loading src:', url);
            if (this.waveSurfer && url) {
                this.waveSurfer.load(url).catch((err) => {
                    console.error('Failed to load audio:', err);
                });
            }
        });
    }

    ngOnInit(): void {
        this.waveSurfer = WaveSurfer.create({
            container: this.waveformRef.nativeElement,
            waveColor: '#d3d3d3',
            progressColor: '#007bff',
            height: 60,
            barWidth: 2,
            cursorWidth: 4,
            interact: true,
        });

        this.waveSurfer.load(this.src());

        this.waveSurfer.on('finish', () => {
            this.isPlaying = false;
            this.currentTime = 0;
        });

        // Update timestamp every 200ms
        this.intervalId = setInterval(() => {
            if (this.waveSurfer && this.isPlaying) {
                this.currentTime = this.waveSurfer.getCurrentTime();
            }
        }, 200);
    }

    togglePlay(): void {
        this.waveSurfer.playPause();
        this.isPlaying = !this.isPlaying;

        // Emit event only when starting playback
        if (this.isPlaying) {
            this.playEvent.emit();
        }
    }

    ngOnDestroy(): void {
        this.waveSurfer.destroy();
        clearInterval(this.intervalId);
    }

    // Helper function to format seconds into mm:ss
    formatTime(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${this.pad(mins)}:${this.pad(secs)}`;
    }

    private pad(num: number): string {
        return num < 10 ? '0' + num : '' + num;
    }
}
