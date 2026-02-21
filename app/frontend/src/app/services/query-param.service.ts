import { Injectable } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({ providedIn: 'root' })
export class QueryParamService {
    constructor(private router: Router, private route: ActivatedRoute) { }

    // Watch a query param as an observable
    watchParam<T = string>(key: string, fallback?: T): Observable<T> {
        return this.route.queryParams.pipe(
            map(params => (params[key] !== undefined ? (params[key] as T) : fallback as T))
        );
    }

    // Get current value once
    getParam<T = string>(key: string, fallback?: T): T {
        const params = this.route.snapshot.queryParams;
        return params[key] !== undefined ? (params[key] as T) : (fallback as T);
    }

    // Set/merge a query param
    setParam(key: string, value: any) {
        this.router.navigate([], {
            queryParams: { [key]: value },
            queryParamsHandling: 'merge',
        });
    }

    // Remove a query param
    removeParam(key: string) {
        console.log(`Removing query param: ${key}`);
        const params = { ...this.route.snapshot.queryParams };
        delete params[key];
        this.router.navigate([], { queryParams: params });
    }

    removeParams(keys: string[]) {
        const params = { ...this.route.snapshot.queryParams };
        keys.forEach(key => delete params[key]);
        this.router.navigate([], { queryParams: params });
    }
}
