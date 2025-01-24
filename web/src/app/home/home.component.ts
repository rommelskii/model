import { Component } from '@angular/core';
import { NavigationService } from '../navigation.service';
import { HttpClient } from '@angular/common/http';
import { HttpService } from '../http.service';
import { OnInit } from '@angular/core';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})


export class HomeComponent {
  constructor(private navigation:NavigationService, private http:HttpService) {}

  ngOnInit(): void {
    this.navigation.navigateTo('/home');
  }
  
  onNavigate(): void {
    this.navigation.navigateTo('/capture');
  }

  testForConnectivity() {
    this.http.getPosts().subscribe({
      next: (response) => {
        alert(response.status)
      }, 
      error: (error) => {
        alert('error, see console');
        console.error('Error: ', error);
      }
    })
  }
}
