/*
  CAPTURE COMPONENT
*/
import { Component, OnInit } from '@angular/core';
import { WebcamComponent } from '../webcam/webcam.component';
import { HttpService } from '../http.service';
import { NavigationService } from '../navigation.service';
import { ViewChild } from '@angular/core';
import { ElementRef } from '@angular/core';

@Component({
  selector: 'app-capture',
  imports: [WebcamComponent],
  templateUrl: './capture.component.html',
  styleUrl: './capture.component.scss'
})
export class CaptureComponent {
  @ViewChild(WebcamComponent) webcam!: WebcamComponent;
  constructor(private http:HttpService, private navigation:NavigationService) {}
  imageStatus = 0;
  cancelDisplay = 'none';
  image1: string = '';
  image2: string = '';

  onImageCapture(value:string): void {
    if (this.imageStatus === 0) {
      this.image1 = value;
      this.imageStatus += 1;
      this.webcam.showCancel();
    } else if (this.imageStatus === 1) {
      this.image2 = value;
      this.imageStatus = 2;
      this.webcam.showCancel();
    } else if (this.imageStatus === 2) {
      this.sendPayload();
      this.imageStatus = 0;
      this.webcam.hideCancel();
    }
  }

  sendPayload(): void {
    //const payload = {'image1': this.image1, 'image2': this.image2};
    const payload = {"image": this.image1};
    this.navigation.navigateTo('loading');
    this.http.postData(payload).subscribe(
      (response) => {
        console.log(response);
        this.navigation.navigateTo('finished');
        this.webcam.stopCamera();
      },
      (error) => {
        console.error('Error: ', error);
      }
    )
  }

  onBack(): void {
    this.navigation.navigateTo('/home');
    this.webcam.stopCamera();
  }

  onCancel(): void {
    this.imageStatus = 0;
    this.webcam.hideCancel();
  }
}
