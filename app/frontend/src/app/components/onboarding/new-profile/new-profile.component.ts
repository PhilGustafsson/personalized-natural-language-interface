import { Component, EventEmitter, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';

@Component({
    selector: 'app-new-profile',
    standalone: true,
    templateUrl: './new-profile.component.html',
    imports: [FormsModule, CommonModule, ButtonModule, InputTextModule]
})
export class NewProfileComponent {
    @Output() ok = new EventEmitter<string>();
    @Output() cancel = new EventEmitter<void>();

    /** Username entered by the user. Required to create profile. */
    username: string = '';

    /** Error message to display if profile creation fails. */
    errorMessage: string = '';

    onSubmit(): void {
        // Clear previous error
        this.errorMessage = '';
        // Emit username to parent (onboarding-page)
        this.ok.emit(this.username);
    }

    setError(message: string): void {
        this.errorMessage = message;
    }
}