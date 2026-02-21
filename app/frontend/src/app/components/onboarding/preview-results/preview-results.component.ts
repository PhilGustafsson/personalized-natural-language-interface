import { Component, EventEmitter, Output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { MicCircleMeterComponent } from "../../shared/mic-circle-meter/mic-circle-meter.component";
import { AudioPlayerComponent } from "../../shared/audio-player/audio-player.component";
import { TtsInteractiveTestComponent } from "./tts-interactive-test/tts-interactive-test.component";
import { TtsService } from '../../../services/tts.service';
import { SttInteractiveTestComponent } from "./stt-interactive-test/stt-interactive-test.component";

@Component({
    selector: 'app-preview-results',
    standalone: true,
    templateUrl: './preview-results.component.html',
    imports: [CommonModule, ButtonModule, TtsInteractiveTestComponent, SttInteractiveTestComponent],
    providers: [TtsService]
})
export class PreviewResultsComponent {

    @Output() onCancel = new EventEmitter<void>();
    @Output() onEndPersonalization = new EventEmitter<void>();
    @Output() onGoToEvaluation = new EventEmitter<void>();

    constructor(ttsService: TtsService) {
        ttsService.setSettings({
            model: 'fineTuned',
            useProfileService: true
        });
    }
 
    deleteProfile() {
        console.log("Deleting profile...");
        this.onCancel.emit();
    }

    endPersonalization() {
        console.log("Ending personalization...");
        this.onEndPersonalization.emit();
    }

    goToEvaluation() {
        console.log("Going to evaluation...");
        this.onGoToEvaluation.emit();
    }
}