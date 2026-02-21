import { Component, EventEmitter, Output } from '@angular/core';
import { ProfileService } from '../../../services/profile.service';
import { PromptType1Component } from "./prompt-type1/prompt-type1-component";

@Component({
    selector: 'app-main-prompt-view',
    standalone: true,
    templateUrl: './main-prompt-view.component.html',
    imports: [PromptType1Component]
})
export class MainPromptViewComponent {

    // Event emitted when all prompting tasks are completed.
    @Output() onCompleted = new EventEmitter<void>();

    constructor(protected profileService: ProfileService) {
        
    }

    completeCurrentPromptTask() {
        console.log("Prompting completed!");
        this.onCompleted.emit();
    }
}
