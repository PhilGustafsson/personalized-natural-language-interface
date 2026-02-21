import { Component, Signal, signal, ViewChild } from '@angular/core';
import { NewProfileComponent } from './new-profile/new-profile.component';
import { ActivatedRoute, Router } from '@angular/router';
import { ProfileCreatedComponent } from "./profile-created/profile-created.component";
import { ProfileService } from '../../services/profile.service';
import { DownloadModelComponent } from "./download-model/download-model.component";
import { AudioRecorderComponent } from './audio-recorder/audio-recorder.component';
import { ChooseDomainComponent } from './choose-domain/choose-domain-component';
import { GeneratePromptsComponent } from './generate-prompts/generate-prompts.component';
import { MainPromptViewComponent } from "./prompting/main-prompt-view.component";
import { toSignal } from '@angular/core/rxjs-interop';
import { map } from 'rxjs';
import { FineTuningComponent } from "./fine-tuning/fine-tuning.component";
import { EvaluationComponent } from '../evaluation/evaluation.component';
import { ButtonModule } from 'primeng/button';
import { PromptService } from '../../services/prompt.service';

/**
 * The root page of the onboarding flow.
 */
@Component({
    selector: 'app-onboarding-page',
    standalone: true,
    templateUrl: './onboarding-page.component.html',
    imports: [NewProfileComponent, ProfileCreatedComponent, DownloadModelComponent, AudioRecorderComponent, MainPromptViewComponent, FineTuningComponent, EvaluationComponent, ChooseDomainComponent, GeneratePromptsComponent, ButtonModule],
    providers: [ProfileService]
})
export class OnboardingPageComponent {
    readonly OnboardingSteps = OnboardingSteps;
    readonly currentStep!: Signal<OnboardingSteps>; //= signal<OnboardingSteps>(OnboardingSteps.CreateProfile);

    @ViewChild('newProfileRef') newProfileRef?: NewProfileComponent;

    constructor(
        private router: Router,
        private route: ActivatedRoute,
        protected profileService: ProfileService,
        private promptService: PromptService,
    ) 
    {
        this.currentStep = toSignal(this.route.queryParams.pipe(
            map(params => params['onboardingStep'] ? +params['onboardingStep'] : 1)
        ), {initialValue: 1});
    }

    cancel(forceQuit = false) {
        console.log('cancel onboarding process');

        // Todo: show dialog if forceQuit = false.

        // Delete profile from backend.
        if (this.profileService.hasProfile) {
            this.profileService.deleteProfile().subscribe({
                next: res => console.log(res.message),
                error: err => console.error('Error deleting profile:', err)
            });
        }

        this.router.navigate(['/']);
    }

    // Accept username from the new-profile component when creating profile
    nextStep(username?: string) {
        switch (this.currentStep()) {
            case OnboardingSteps.CreateProfile:
                // todo: display loading indicator while waiting for response.
                this.profileService.newProfile(username).subscribe({
                    next: (success: any) => {
                        if (success) {
                            this.setStep(OnboardingSteps.ProfileCreated);
                        }
                        else {
                            console.error('Failed to create profile');
                        }
                    },
                    error: (err: any) => {
                        // Display error to user via the new-profile component
                        if (this.newProfileRef) {
                            const errorMsg = err.message || 'Failed to create profile. Please try again.';
                            this.newProfileRef.setError(errorMsg);
                        }
                        console.error('Error creating profile:', err);
                    }
                });
                break;
            
            case OnboardingSteps.ProfileCreated:
                this.profileService.areModelsDownloaded().subscribe(isDownloaded => {
                    if (isDownloaded) {
                        this.setStep(OnboardingSteps.MicCheck);
                    }
                    else {
                        this.setStep(OnboardingSteps.DownloadModel);
                    }
                });
                this.setStep(OnboardingSteps.DownloadModel);
                break;
            
            case OnboardingSteps.DownloadModel:
                this.setStep(OnboardingSteps.MicCheck);
                break;

            case OnboardingSteps.MicCheck:
                this.setStep(OnboardingSteps.ChooseDomain);
                break;
            
            case OnboardingSteps.Prompt:
                this.setStep(OnboardingSteps.FineTune);
                break;

            case OnboardingSteps.FineTune:
                this.setStep(OnboardingSteps.Evaluation);
                break;

            case OnboardingSteps.Evaluation:
                this.endPersonalization();
                break;
        }
    }

        onDomainSelected(sel: string) {
            if (sel === 'prompt') {
                // Load predefined Swedish prompts
                this.promptService.loadSvStandardPrompts();
                this.setStep(OnboardingSteps.Prompt);
            } else if (sel === 'generate') {
                this.setStep(OnboardingSteps.GeneratePrompts);
            }
        }

    setStep(onboardingStep: OnboardingSteps) {
        this.router.navigate([], {
            queryParams: {onboardingStep},
            queryParamsHandling: 'merge'
        });
    }

    onGeneratePromptsComplete() {
        // After prompts are generated and loaded, go to prompt recording
        this.setStep(OnboardingSteps.Prompt);
    }

    onGeneratePromptsCancel() {
        this.setStep(OnboardingSteps.ChooseDomain);
    }

    endPersonalization() {
        // todo: tell the backend that profile is complete?
        // todo: clear query params?
        this.router.navigate(['/']);
    }

    goToEvaluation() {
        this.router.navigate(['/evaluation']);
    }
    
}

enum OnboardingSteps {
    CreateProfile = 1,
    ProfileCreated,
    DownloadModel,
    MicCheck,
    ChooseDomain,
    GeneratePrompts,
    Prompt,
    FineTune,
    Evaluation
}