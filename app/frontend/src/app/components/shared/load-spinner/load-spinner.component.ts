import { NgClass } from '@angular/common';
import { Component, input, Input } from '@angular/core';

@Component({
    selector: 'app-load-spinner',
    template: `
        <div class="spinner" [ngClass]="colorTheme()"></div>
    `,
    styles: [`
        .spinner {
            border: 3px solid white;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            min-width: 20px;
            min-height: 20px;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin: auto;
        }

        .dark {
            border: 3px solid #007bff;
            border-top: 3px solid white;
        }

        .grayed-out {
            border: 3px solid #e0e0e0;
            border-top: 3px solid white;
        }

        @keyframes spin {
        to { transform: rotate(360deg); }
        }    
    `],
    imports: [NgClass]
})
export class LoadSpinnerComponent {
    readonly colorTheme = input('light'); // values: 'light', 'dark', 'grayed-out'
}