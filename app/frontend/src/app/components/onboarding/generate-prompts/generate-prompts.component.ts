import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { LLMService } from '../../../services/llm.service';
import { PromptService } from '../../../services/prompt.service';

@Component({
  selector: 'app-generate-prompts',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ButtonModule
  ],
  templateUrl: './generate-prompts.component.html',
  styleUrls: ['./generate-prompts.component.scss']
})
export class GeneratePromptsComponent {
  @Output() promptsGenerated = new EventEmitter<void>();
  @Output() cancel = new EventEmitter<void>();

  // Generation parameters
  domain: string = '';
  numPrompts: number = 20;
  language: string = 'sv';
  difficulty: string = 'intermediate';
  sentenceLength: string = 'medium';
  includeTechnicalTerms: boolean = true;
  style: string = 'conversational';

  // UI state
  isGenerating: boolean = false;
  error: string = '';
  generatedPrompts: string[] = [];
  showPreview: boolean = false;

  // Dropdown options
  languageOptions = [
    { label: 'Swedish', value: 'sv' },
    { label: 'English', value: 'en' }
  ];

  difficultyOptions = [
    { label: 'Beginner', value: 'beginner' },
    { label: 'Intermediate', value: 'intermediate' },
    { label: 'Advanced', value: 'advanced' }
  ];

  sentenceLengthOptions = [
    { label: 'Short (5-10 words)', value: 'short' },
    { label: 'Medium (10-20 words)', value: 'medium' },
    { label: 'Long (20+ words)', value: 'long' }
  ];

  styleOptions = [
    { label: 'Conversational', value: 'conversational' },
    { label: 'Formal', value: 'formal' },
    { label: 'Technical', value: 'technical' }
  ];

  // Domain suggestions
  domainSuggestions = [
    'AI and machine learning',
    'Medicine and healthcare',
    'Finance and banking',
    'Legal and law',
    'Customer service',
    'Software engineering',
    'Marketing and sales',
    'Education',
    'Gaming',
    'Science and research'
  ];

  constructor(
    private llmService: LLMService,
    private promptService: PromptService
  ) {}

  selectDomainSuggestion(domain: string) {
    this.domain = domain;
  }

  generatePrompts() {
    if (!this.domain.trim()) {
      this.error = 'Please enter a domain';
      return;
    }

    this.isGenerating = true;
    this.error = '';

    this.llmService.generatePrompts({
      domain: this.domain,
      num_prompts: this.numPrompts,
      language: this.language,
      difficulty: this.difficulty as any,
      sentence_length: this.sentenceLength as any,
      include_technical_terms: this.includeTechnicalTerms,
      style: this.style as any
    }).subscribe({
      next: (prompts) => {
        this.generatedPrompts = prompts;
        this.showPreview = true;
        this.isGenerating = false;
        this.error = '';
      },
      error: (err) => {
        console.error('Failed to generate prompts:', err);
        this.error = err.error?.message || err.error?.error || 'Failed to generate prompts. Please try again.';
        this.isGenerating = false;
      }
    });
  }

  usePrompts() {
    // Load generated prompts into prompt service
    console.log('Loading generated prompts into prompt service:', this.generatedPrompts.length, 'prompts');
    console.log('First 3 prompts:', this.generatedPrompts.slice(0, 3));
    this.promptService.setPrompts(this.generatedPrompts);
    console.log('Prompts loaded, emitting promptsGenerated event');
    
    // Small delay to ensure signals update before navigation
    setTimeout(() => {
      console.log('Emitting promptsGenerated after delay');
      this.promptsGenerated.emit();
    }, 100);
  }

  regenerate() {
    this.showPreview = false;
    this.generatePrompts();
  }

  onCancel() {
    this.cancel.emit();
  }
}
