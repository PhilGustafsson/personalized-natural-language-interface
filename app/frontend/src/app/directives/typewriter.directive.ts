// typewriter.directive.ts
import { Directive, ElementRef, Input, Renderer2, effect, signal } from '@angular/core';

@Directive({
  selector: '[appTypewriter]',
  standalone: true
})
export class TypewriterDirective {
  @Input({ required: true }) appTypewriter!: string;
  @Input() typingSpeed = 40;
  @Input() punctuationPause = 200;

  private currentText = signal('');
  private typingDone = signal(false);

  constructor(private el: ElementRef<HTMLElement>, private renderer: Renderer2) {
    // Update DOM as text changes
    effect(() => {
      this.el.nativeElement.textContent = this.currentText();

      // Add/remove blinking class
      if (this.typingDone()) {
        this.renderer.removeClass(this.el.nativeElement, 'typewriter-typing');
      } else {
        this.renderer.addClass(this.el.nativeElement, 'typewriter-typing');
      }
    });
  }

  ngOnChanges() {
    if (this.appTypewriter) {
      this.typeOutText(this.appTypewriter);
    }
  }

  private async typeOutText(text: string) {
    this.typingDone.set(false);
    this.currentText.set('');

    for (let i = 0; i < text.length; i++) {
      this.currentText.update(t => t + text[i]);
      const char = text[i];
      const delay =
        [',', '.', '!', '?'].includes(char)
          ? this.typingSpeed + this.punctuationPause
          : this.typingSpeed;

      await new Promise(res => setTimeout(res, delay));
    }

    // Finished typing
    this.typingDone.set(true);
  }
}
