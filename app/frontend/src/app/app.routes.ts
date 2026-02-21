import { Routes } from '@angular/router';
import { MainMenuComponent } from './components/main-menu/main-menu.component';

export const routes: Routes = [
    { path: '', component: MainMenuComponent },
    { path: 'new-profile', loadComponent: () => 
        import('./components/onboarding/onboarding-page.component').then(m => m.OnboardingPageComponent)},
    { path: 'delete-profile', loadComponent: () =>
        import('./components/onboarding/delete-profile/delete-profile.component').then(m => m.DeleteProfileComponent) 
    },
    { path: 'choose-profile', loadComponent: () =>
        import('./components/onboarding/choose-profile/choose-profile.component').then(m => m.ChooseProfileComponent)
    },
    { path: 'stt', loadComponent: () =>
        import('./components/shared/stt-manager/stt-manager.component').then(m => m.SttManagerComponent)
    },
    { path: 'evaluation', loadComponent: () => import('./components/evaluation/evaluation.component').then(m => m.EvaluationComponent) }
];