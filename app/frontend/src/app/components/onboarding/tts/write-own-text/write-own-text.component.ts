import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TtsInteractiveTestComponent } from '../../preview-results/tts-interactive-test/tts-interactive-test.component';

@Component({
    selector: 'app-write-own-text',
    standalone: true,
    templateUrl: './write-own-text.component.html',
    imports: [CommonModule, TtsInteractiveTestComponent]
})
export class WriteOwnTextComponent {
}
