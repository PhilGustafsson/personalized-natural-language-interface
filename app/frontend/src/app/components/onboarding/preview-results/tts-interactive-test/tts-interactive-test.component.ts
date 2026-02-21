import { Component, computed, effect, EventEmitter, Input, input, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { TextareaModule } from 'primeng/textarea';
import { AudioPlayerComponent } from "../../../shared/audio-player/audio-player.component";
import { TTSResults, TtsService } from '../../../../services/tts.service';
import { FormsModule } from '@angular/forms';

export interface ComponentInputData {
    text: string;
    ttsResult: TTSResults;
}

@Component({
    selector: 'app-tts-interactive-test',
    templateUrl: './tts-interactive-test.component.html',
    imports: [CommonModule, ButtonModule, TextareaModule, AudioPlayerComponent, FormsModule]
})
export class TtsInteractiveTestComponent implements OnInit {
    inputData = input<ComponentInputData | null>(null);

    protected text = '';
    protected readonly ttsResult = signal<TTSResults | null>(null);
    protected readonly audio = computed(() => this.ttsResult()?.audioBlob || null);
    protected readonly recordingURL = computed(() => this.ttsResult()?.audioURL || null);

    protected readonly isProcessingText = signal(false);

    readonly onClickConvert = new EventEmitter<string>();
    readonly onTextConverted = new EventEmitter<TTSResults>();

    constructor(
        private ttsService: TtsService
    ) {
        effect(() => console.log('recordingURL: ', this.recordingURL()));
    }

    ngOnInit(): void {
        if (this.inputData() != null) {
            this.text = this.inputData()!.text;
            this.ttsResult.set(this.inputData()!.ttsResult);
        }
    }

    convertToSpeech() {
        console.log("Speaking text: " + this.text);

        this.onClickConvert.emit(this.text);

        if (this.text == '') {
            // do nothing.
            console.log("No text to speak.");
            return;
        }

        // Start processing text.
        this.isProcessingText.set(true);
        this.ttsService.toAudio(this.text).subscribe({
            next: results => {
                console.log("Received audio blob from TTS.");
                this.isProcessingText.set(false);
                this.ttsResult.set(results);
            },
            error: err => {
                console.error("Error in TTS: ", err);
                this.isProcessingText.set(false);
            }
        });
    }

}