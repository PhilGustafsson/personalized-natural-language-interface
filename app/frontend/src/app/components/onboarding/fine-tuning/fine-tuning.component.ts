import { Component, computed, EventEmitter, OnInit, Output, Signal, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { ProgressBarModule } from 'primeng/progressbar';
import { FineTuningResponse, FineTuningService } from '../../../services/fine-tuning.service';
import { ProfileService } from '../../../services/profile.service';
import { QueryParamService } from '../../../services/query-param.service';
import { takeUntilDestroyed, toObservable, toSignal } from '@angular/core/rxjs-interop';
import { of, switchMap, tap } from 'rxjs';
import { SttService } from '../../../services/stt.service';

@Component({
    selector: 'app-fine-tuning',
    standalone: true,
    templateUrl: './fine-tuning.component.html',
    imports: [CommonModule, ButtonModule, ProgressBarModule]
})
export class FineTuningComponent implements OnInit {
    
    @Output() ok = new EventEmitter<void>();
    @Output() cancel = new EventEmitter<void>();

    private readonly jobId_TTS = signal<string | null>(null);
    private readonly jobId_STT = signal<string | null>(null);

    protected readonly isFinetuningTTS = signal(true);
    protected readonly isFinetuningSTT = signal(true);
    
    private modelSelected = false;

    protected readonly statusTTS: Signal<FineTuningResponse | null>;
    protected readonly statusSTT: Signal<FineTuningResponse | null>;
    protected readonly isFineTuningComplete: Signal<boolean>;
    protected readonly timeRemaining: Signal<number>; // in seconds.
    protected readonly progressTTS: Signal<number>; // 0-100.
    protected readonly progressSTT: Signal<number>; // 0-100.

    constructor(
        protected fineTuningService: FineTuningService,
        protected profileService: ProfileService,
        protected queryParamService: QueryParamService,
        protected sttService: SttService
    ) {
        // Get status for TTS.
        this.statusTTS = toSignal(toObservable(this.jobId_TTS).pipe(
            takeUntilDestroyed(),
            switchMap(id => {
                if (id != null) return this.fineTuningService.pollFineTuningStatus(id);
                else return of(null);
            })
        ), {initialValue: null});

        // Get status for STT.
        this.statusSTT = toSignal(toObservable(this.jobId_STT).pipe(
            takeUntilDestroyed(),
            switchMap(id => {
                if (id != null) return this.fineTuningService.pollFineTuningStatus(id);
                else return of(null);
            }),
            tap(status => {
                if (status?.result && status.progress === 100 && !this.modelSelected) {
                    this.modelSelected = true;
                    this.selectNewModel(status.result.finetuned_dir);
                }
            })
        ), {initialValue: null});

        // Update progress and time remaining when status changes.
        this.progressTTS = computed(() => this.statusTTS()?.progress ?? 0);
        this.progressSTT = computed(() => this.statusSTT()?.progress ?? 0);
        this.timeRemaining = computed(() => {
            const tts = this.statusTTS()?.estimatedTimeRemaining ?? 0;
            const stt = this.statusSTT()?.estimatedTimeRemaining ?? 0;
            return Math.max(tts, stt);
        });

        this.isFineTuningComplete = computed(() => {
            const ttsComplete = !this.isFinetuningTTS() || (this.statusTTS()?.progress === 100);
            const sttComplete = !this.isFinetuningSTT() || (this.statusSTT()?.progress === 100);
            return ttsComplete && sttComplete;
        });
    }

    ngOnInit() {
        // Retrieve the TTS job ID.
        if (this.isFinetuningTTS()) {
            const jobIdTTS = this.queryParamService.getParam<string | null>('jobIdTTS', null);
            if (jobIdTTS == null) {
                this.startFineTuningTTS();
            }
            else {
                this.jobId_TTS.set(jobIdTTS);
            }
        }

        // Retrieve the STT job ID.
        if (this.isFinetuningSTT()) {
            const jobIdSTT = this.queryParamService.getParam<string | null>('jobIdSTT', null);
            if (jobIdSTT == null) {
                this.startFineTuningSTT();
            }
            else {
                this.jobId_STT.set(jobIdSTT);
            }
        }
    }

    private selectNewModel(modelDir: string) {
        console.log("Selecting new model:", modelDir);
        this.sttService.selectModel({
            model_dir: modelDir,
            whisper_language: 'Swedish' // Assuming Swedish for now, or get from profile
        }).subscribe({
            next: () => console.log("Model selected successfully"),
            error: err => console.error("Failed to select model:", err)
        });
    }

    private startFineTuningSTT() {
        console.log('Starting fine-tuning STT...');
        const profileID = this.profileService.profileID()!;
        this.fineTuningService.startFineTuningSTT({profileID: profileID}).subscribe({
            next: response => {
                this.jobId_STT.set(response.jobId);
                this.queryParamService.setParam('jobIdSTT', response.jobId);
            },
            error: err => console.error('Error starting fine-tuning STT:', err)
        });
    }

    private startFineTuningTTS() {
        console.log('Starting fine-tuning TTS...');
        const profileID = this.profileService.profileID()!;
        this.fineTuningService.startFineTuningTTS({profileID: profileID}).subscribe({
            next: response => {
                this.jobId_TTS.set(response.jobId);
                this.queryParamService.setParam('jobIdTTS', response.jobId);
            },
            error: err => console.error('Error starting fine-tuning TTS:', err)
        });
    }

    cancelFineTuning() {
        console.log('Cancelling fine-tuning...');
        if (this.isFinetuningSTT() && this.jobId_STT() != null) {
            this.fineTuningService.cancelFineTuning(this.jobId_STT()!).subscribe();
        }
        if (this.isFinetuningTTS() && this.jobId_TTS() != null) {
            this.fineTuningService.cancelFineTuning(this.jobId_TTS()!).subscribe();
        }

        this.queryParamService.removeParams(['jobIdSTT', 'jobIdTTS']);

        // Wait a bit to let query params update before emitting event.
        setTimeout(() => {
            this.cancel.emit();
        }, 100);
    }

    continue() {
        console.log('Fine-tuning complete, continuing...');
        this.queryParamService.removeParams(['jobIdSTT', 'jobIdTTS']);

        // Wait a bit to let query params update before emitting event.
        setTimeout(() => {
            this.ok.emit();
        }, 100);
    }
}