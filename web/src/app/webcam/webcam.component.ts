import { Component, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-webcam',
  templateUrl: './webcam.component.html',
  styleUrls: ['./webcam.component.scss']
})
export class WebcamComponent implements AfterViewInit {
  @ViewChild('video') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvasElement!: ElementRef<HTMLCanvasElement>;
  @ViewChild('cancelButton') cancelButton!: ElementRef<HTMLButtonElement>;
  @Output() imageCaptured = new EventEmitter<string>();
  @Output() cancelEvent = new EventEmitter<void>();

  private stream: MediaStream | null = null;  // To hold the stream

  ngAfterViewInit(): void {
    this.startCamera();
    this.hideCancel();  // Hide cancel button after view initialization
  }

  // Starts the webcam by selecting the first video input device
  startCamera(): void {
    navigator.mediaDevices.enumerateDevices()
      .then((devices) => {
        // Filter to get only video input devices
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        if (videoDevices.length > 0) {
          // Select the first available video device (you can change this logic if needed)
          const videoDeviceId = videoDevices[0].deviceId;
          
          // Get the media stream from the selected device
          navigator.mediaDevices.getUserMedia({ video: { deviceId: videoDeviceId } })
            .then((stream) => {
              this.stream = stream;  // Store the stream to stop later
              const video = this.videoElement.nativeElement;
              video.srcObject = stream;
              video.play();
            })
            .catch((err) => {
              console.error('Error accessing webcam:', err);
            });
        } else {
          console.error('No video input devices found.');
        }
      })
      .catch((err) => {
        console.error('Error enumerating devices:', err);
      });
  }

  // Stops the webcam
  stopCamera(): void {
    if (this.stream) {
      const tracks = this.stream.getTracks();
      tracks.forEach((track) => track.stop());  // Stop all tracks (video, audio)
      this.stream = null;  // Optionally nullify the stream
      console.log('Camera stopped');
    }
  }

  // Show the cancel button
  showCancel(): void {
    if (this.cancelButton) {
      this.cancelButton.nativeElement.style.display = 'inline';
    }
  }

  // Hide the cancel button
  hideCancel(): void {
    if (this.cancelButton) {
      this.cancelButton.nativeElement.style.display = 'none';
    }
  }

  // Emits the cancel event to the parent component
  initCancel(): void {
    this.cancelEvent.emit();
  }

  // Captures an image from the webcam feed
  captureImage(): void {
    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement.nativeElement;
    const context = canvas.getContext('2d');
  
    if (context) {
      // Set canvas dimensions to match video feed
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
  
      // Draw the current frame from the video onto the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
  
      // Convert canvas content to a Base64 image string
      const capturedImage = canvas.toDataURL('image/png');
  
      // Remove the "data:image/png;base64," prefix
      const base64Image = capturedImage.split(',')[1];
  
      // Emit only the raw Base64 string
      this.imageCaptured.emit(base64Image);
    }
  }
  
}
