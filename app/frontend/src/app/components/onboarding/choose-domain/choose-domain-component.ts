import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-choose-domain',
  standalone: true,
  imports: [CommonModule, ButtonModule],
  templateUrl: './choose-domain-component.html'
})
export class ChooseDomainComponent {
  @Output() selected = new EventEmitter<string>();

  constructor(private router: Router) {}

  chooseReadPrompts() {
    // Navigate to onboarding prompt step
    this.selected.emit('prompt');
  }

  chooseGeneratePrompts() {
    // Navigate to custom prompt generation
    this.selected.emit('generate');
  }
}
