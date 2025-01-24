import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HttpService {

  constructor(private http:HttpClient) { }

  getPosts(): Observable<any> {
    return this.http.get<any>('http://0.0.0.0:6969/');
  }

  postData(payload: any): Observable<any> {
    const headers = new HttpHeaders().set('Content-Type', 'application/json');
    return this.http.post<any>('http://0.0.0.0:6969/process', payload, { headers })
  }
}
