import { Routes } from '@angular/router';
import { CaptureComponent } from './capture/capture.component';
import { HomeComponent } from './home/home.component';
import { LoadingComponent } from './loading/loading.component';
import { FinishedComponent } from './finished/finished.component';

export const routes: Routes = [
    { path: '', component: HomeComponent },
    { path: 'home', component: HomeComponent },
    { path: 'capture', component: CaptureComponent },
    { path: 'loading', component: LoadingComponent },
    { path: 'finished', component: FinishedComponent }
];
