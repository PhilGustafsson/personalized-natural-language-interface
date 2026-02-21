import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { WriteOwnTextComponent } from './write-own-text/write-own-text.component';
import { EvaluateAudioComponent } from './evaluate-audio/evaluate-audio.component';
import { ButtonModule } from 'primeng/button';

@Component({
    selector: 'app-tts-menu',
    standalone: true,
    templateUrl: './tts-menu.component.html',
    imports: [CommonModule, WriteOwnTextComponent, EvaluateAudioComponent, ButtonModule]
})
export class TtsMenuComponent {
    @Output() close = new EventEmitter<void>();

    mode: 'menu' | 'write' | 'evaluate' = 'menu';

    constructor(private router: Router) {}

    startWrite() { this.mode = 'write'; }
    startEvaluate() { this.mode = 'evaluate'; }
    goBack() { this.mode = 'menu'; this.close.emit(); }
    cancelToMain() { this.router.navigate(['/']); }
}
