import { Component, computed, effect, EventEmitter, Input, Output, Signal, signal } from '@angular/core';
import { AudioPlayerComponent } from "../../../shared/audio-player/audio-player.component";
import { RecordingService } from '../../../../services/recording.service';
import { AudioQualityService, Quality } from '../../../../services/audio-quality.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { SecondsToMmssPipe } from '../../../../pipes/seconds-to-mmss.pipe';
import { MicLevelService } from '../../../../services/mic-level.service';
import { MicCircleMeterComponent } from "../../../shared/mic-circle-meter/mic-circle-meter.component";
import { PromptService } from '../../../../services/prompt.service';
import { LLMService } from '../../../../services/llm.service';
import { FineTuningService } from '../../../../services/fine-tuning.service';
import { ProfileService } from '../../../../services/profile.service';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { ProgressBarModule } from 'primeng/progressbar';

enum ViewState {
    Prompt = 1,
    Training
}

enum PromptState {
    LoadingPrompt,
    Ready,
    Recording,
    AutomaticQualityCheck,
    ManualQualityCheck
}

@Component({
    selector: 'app-prompt-type1',
    standalone: true,
    templateUrl: './prompt-type1-component.html',
    imports: [AudioPlayerComponent, SecondsToMmssPipe, MicCircleMeterComponent, FormsModule, CommonModule, ButtonModule, InputTextModule, ProgressBarModule]
})
export class PromptType1Component {

    @Input() manualCheckOption = "always"; // can be "always", "onFailedAutoCheck", "never"

    // Event emitted when all prompts are completed.
    @Output() onCompleted = new EventEmitter<void>();
    
    protected ViewState = ViewState;
    protected currentViewState = signal(ViewState.Prompt);

    protected PromptState = PromptState;
    protected currentPromptState = signal(PromptState.Ready);
    protected isRecording = computed(() => this.currentPromptState() == PromptState.Recording);

    protected readonly promptText!: Signal<string>;
    protected readonly promptNumber!: Signal<number>;
    protected readonly totalPrompts!: Signal<number>;
    
    protected userCheckedAudio = false; // Did the user actually play the audio?
    protected automaticCheckQuality = signal<Quality | null>(null);
    protected passAutomaticCheck = computed(() => this.automaticCheckQuality()?.passed ?? false);
    private recording = signal<Blob | null>(null); // The current recording.

    protected readonly recordingURL = computed(() => {
        const rec = this.recording();
        return rec != null ? this.recordingService.createAudioURL(rec) : null;
    });
    protected readonly recordingTime: Signal<number>;
    protected readonly micLevel: Signal<number>;
    
    // Training state
    protected isTrainingStarted = signal(false);
    protected trainingMessage = signal<string | null>(null);
    protected errorMessage = signal<string | null>(null); // For training errors

    constructor(
        private recordingService: RecordingService,
        private audioQualityService: AudioQualityService,
        private micLevelService: MicLevelService,
        private promptService: PromptService,
        private llmService: LLMService,
        private fineTuningService: FineTuningService,
        private profileService: ProfileService,
        private router: Router,
        private route: ActivatedRoute,
    ) {
        this.recordingTime = toSignal(recordingService.getElapsedTime(), {initialValue: 0});
        this.micLevel = toSignal(micLevelService.getMicLevel(), {initialValue: 0});
        this.promptText = promptService.currentPromptText;
        this.promptNumber = promptService.currentPromptNumber;
        this.totalPrompts = promptService.totalPrompts;
        
        // Complete onboarding when all prompts are done.
        effect(() => {
            if (promptService.promptingComplete()) {
                this.currentViewState.set(ViewState.Training);
            }
        }, {allowSignalWrites: true});

        // Update view state in query params.
        effect(() => {
            const viewState = this.currentViewState();
            router.navigate([], {
                queryParams: {viewState},
                queryParamsHandling: 'merge'
            });
        });
    }

    skipPrompt() {
        console.log('Skipping prompt (testing mode)');
        this.promptService.nextPrompt();
        this.currentPromptState.set(PromptState.Ready);
    }

    startRecording() {
        console.log('start recording');
        this.currentPromptState.set(PromptState.Recording);
        this.userCheckedAudio = false;
        this.micLevelService.start();
        this.recordingService.startRecording().subscribe({
            next: state => {
                console.log('recording state:', state);
            },
            error: msg => {
                console.error('Recording error:', msg);
                this.currentPromptState.set(PromptState.Ready); // reset.
            }
        });
    }

    stopRecording() {
        console.log('stop recording');
        this.currentPromptState.set(PromptState.AutomaticQualityCheck);

        this.micLevelService.stop();
        this.recordingService.stopRecording().subscribe({
            next: blob => {
                this.recording.set(blob);
                // Allow the user to click "Looks Good" by default after a successful recording
                // so they can proceed even if they don't manually play back the audio.
                // This keeps the manual-check flow intact but improves UX for quick runs.
                this.userCheckedAudio = true;
                this.doAutomaticQualityCheck();
            },
            error: msg => {
                console.error('Recording stop error:', msg);
                this.currentPromptState.set(PromptState.Ready); // reset.
            }
        });
    }

    private doAutomaticQualityCheck() {
        console.log('doing automatic quality check');
        this.audioQualityService.checkQuality(this.recording()!).subscribe({
            next: quality => {
                console.log('automatic quality check result:', quality);
                this.automaticCheckQuality.set(quality);

                // Manual check?
                if (this.manualCheckOption == "always") {
                    this.currentPromptState.set(PromptState.ManualQualityCheck);
                }
                else if (!quality.passed && this.manualCheckOption == "onFailedAutoCheck") {
                    this.currentPromptState.set(PromptState.ManualQualityCheck);
                }
                else {
                    this.nextPrompt(); 
                }
            },
            error: msg => {
                console.error('Quality check error:', msg);
                this.currentPromptState.set(PromptState.ManualQualityCheck); // fallback to manual check.
            }
        });
    }

    userPlayAudio() {
        console.log('user played audio');
        this.userCheckedAudio = true;
    }

    userVerifyAudio() {
        console.log('user verified audio');
        this.nextPrompt();
    }

    doRetake() {
        console.log('user wants to retake audio');
        this.currentPromptState.set(PromptState.Ready);
        // Reset the user-checked flag so the user must reconfirm the new recording.
        this.userCheckedAudio = false;
    }

    nextPrompt() {
        console.log('load next prompt');
        this.currentPromptState.set(PromptState.LoadingPrompt);
        this.promptService.saveRecording({
            recording: this.recording()!,
            qualityCheckedByUser: this.userCheckedAudio,
            passedAutomaticQualityCheck: this.passAutomaticCheck(),
            automaticQualityScore: this.automaticCheckQuality()?.score ?? 0,
        }).subscribe({
            next: result => {
                console.log('Saved recording response:', result);
                this.promptService.nextPrompt();
                this.currentPromptState.set(PromptState.Ready);
            },
            error: err => {
                console.error('Error saving recording:', err);
                // todo: handle error properly.
                this.promptService.nextPrompt();
                this.currentPromptState.set(PromptState.Ready);
            }
        });
        
    }

    startTraining() {
        this.isTrainingStarted.set(true);
        this.trainingMessage.set("Starting training...");
        
        const profileId = this.profileService.profileID();
        if (!profileId) {
            this.errorMessage.set("No profile selected.");
            this.isTrainingStarted.set(false);
            return;
        }

        this.fineTuningService.startFineTuningSTT({ profileID: profileId }).subscribe({
            next: (response) => {
                this.trainingMessage.set("Training started successfully! You can continue to the main menu.");
                // Optionally wait a bit then emit onCompleted
                setTimeout(() => {
                    this.onCompleted.emit();
                }, 2000);
            },
            error: (err) => {
                console.error("Training failed", err);
                this.errorMessage.set("Failed to start training: " + (err.error?.error || err.message));
                this.isTrainingStarted.set(false);
            }
        });
    }
}