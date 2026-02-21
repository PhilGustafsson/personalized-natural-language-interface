import { Component, Signal, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { MicCircleMeterComponent } from "../../../shared/mic-circle-meter/mic-circle-meter.component";
import { MicLevelService } from '../../../../services/mic-level.service';
import { RecordingService } from '../../../../services/recording.service';
import { SttService } from '../../../../services/stt.service';
import { TypewriterDirective } from "../../../../directives/typewriter.directive";
import { toSignal } from '@angular/core/rxjs-interop';
import { ProfileService } from '../../../../services/profile.service';

@Component({
    selector: 'app-stt-interactive-test',
    templateUrl: './stt-interactive-test.component.html',
    imports: [CommonModule, ButtonModule, MicCircleMeterComponent, TypewriterDirective]
})
export class SttInteractiveTestComponent {

    protected readonly micLevel: Signal<number>;
    protected readonly isRecording = signal(false);
    protected readonly isProcessing = signal(false);
    protected readonly transcribedText = signal<string | null>(null);

    constructor(
        private micLevelService: MicLevelService,
        private recordingService: RecordingService,
        private sttService: SttService,
        private profileService: ProfileService
    ) {
        this.micLevel = toSignal(micLevelService.getMicLevel(), {initialValue: 0});
    }

    protected startRecording() {
        console.log("Start recording...");
        this.isRecording.set(true);
        this.recordingService.startRecording();
        this.micLevelService.start();
    }

    protected stopRecording() {
        this.isRecording.set(false);
        this.isProcessing.set(true);
        this.micLevelService.stop();

        this.recordingService.stopRecording().subscribe({
            next: audioBlob => {
                this.transcribeRecording(audioBlob);
            },
            error: err => {
                console.error("Error during recording: ", err);
                this.isProcessing.set(false);
            }
        });
    }

    private transcribeRecording(audioBlob: Blob) {
        this.sttService.toText(audioBlob, true, this.profileService.profileID()).subscribe({
            next: result => {
                this.transcribedText.set(result.text);
                this.isProcessing.set(false);
            },
            error: err => {
                console.error("Error during STT: ", err);
                this.isProcessing.set(false);
            }
        });
    }
}