import { Component } from '@angular/core';
import { NavigationService } from '../navigation.service';

@Component({
  selector: 'app-finished',
  imports: [],
  templateUrl: './finished.component.html',
  styleUrl: './finished.component.scss'
})
export class FinishedComponent {

  constructor(private navigation:NavigationService) {}

  goToHome(): void {
    this.navigation.navigateTo("home");
  }
}
