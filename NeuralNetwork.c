#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nrutil.h"


FILE  *V_NeuralNetwork,*time_file, *Spikes;


void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	void (*derivs)(double, double [], double []));

void derivs(double x,double y[],double dydx[]);

double *f, *df, *V, *etas, *X, *U;


// neural network QIF
int main(){

double t;
double S;
double rtot;
const double maxt=1. ;
const double h=0.0001 ; 
const double N = 25000 ; //  recordar poner 200.000
const double H = 0.0;
const double delta = 0.25;
const double I_B = -1.0;
const double J = 15;
const double tau_e = 0.015;
const double PI = 3.14159265358979323846;
const double Vp = 100; 
const double Vr = -100;
double U_0 = 0.2;
double x_0 = 1; 
const double tau_d = 0.2;
const double tau_f = 1.5;
double I_S;

f=dvector(1,3);
df=dvector(1,3);
V = dvector(1,N+1);
X = dvector(1,N+1);
U = dvector(1,N+1);

V_NeuralNetwork=fopen("V.dat","w");

time_file=fopen("time.dat","w");
Spikes=fopen("Spikes.dat","w");



// initial condition

for (int i = 1; i <= N; i++) {
        etas[i] = H + delta*tan((PI/2)*(2*i-N-1)/(N+1));
        V[i] = -100.0;
        X[i] = x_0;
        U[i] = U_0;
    }  
f[1]= 0;
f[2]= x_0;
f[3]= U_0;
 
t=0.0;

	while (t<=maxt){
        S = 0.0;
        for (int i = 1; i <= N; i++) {
            if (V[i]>= Vp){
                if (i<=20000){ // excitatoria
                    S+=J*X[i]*U[i];
                }else{ // inhibitoria
                    S-=J;
                }
              
              V[i] = Vr; 
            }
        
        }

        if ((t >=0.15 && t<= 0.3) || (t >=0.45 && t<= 0.6)){
            I_S = 2;
        }else{
            I_S = 0;
        }

        for (int i = 1; i <= N; i++) { // Neuronas

            /* Numeric scheme */

            f[1] = V[i];
            f[2] = X[i];
            f[3] = U[i];
            double eta = etas[i];
            derivs(t,f,df,eta, I_B, I_S, tau_e, S, N, tau_d, tau_f, U_0);
            rk4(f,df,3,t,h,f,derivs);
            V[i] = df[1];
            X[i] = df[2];
            U[i] = df[3];
            t=t+h; 
                                        
            fprintf(V_NeuralNetwork,"%lf ", t, f[1]); 


        }
        fprintf(V_NeuralNetwork, "\n");

	}




free_dvector(f,1,3);
free_dvector(df,1,3);

return 0;
}





void derivs(double x,double y[],double dydx[],double eta, double I_B, double I_S, double tau_e, double S, double N, double tau_d, double tau_f, double U_0)
{
    dydx[1] = y[1]*y[1] + eta + I_B + I_S + tau_e*S/N;
    dydx[2] = (1-y[2])/ tau_d - y[2]* y[3]*S;
    dydx[3] = (U_0-y[3])/ tau_f - U_0* (1-y[3])*S;
}







void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	void (*derivs)(double, double [], double []))
{
	int i;
	double xh,hh,h6,*dym,*dyt,*yt;

	dym=dvector(1,n);
	dyt=dvector(1,n);
	yt=dvector(1,n);
	hh=h*0.5;
	h6=h/6.0;
	xh=x+hh;
	for (i=1;i<=n;i++) yt[i]=y[i]+hh*dydx[i];
	(*derivs)(xh,yt,dyt);
	for (i=1;i<=n;i++) yt[i]=y[i]+hh*dyt[i];
	(*derivs)(xh,yt,dym);
	for (i=1;i<=n;i++) {
		yt[i]=y[i]+h*dym[i];
		dym[i] += dyt[i];
	/*fprintf(fdue,"%d%.10e %.10e\n",i,dym[i],y[i]);*/
	}
	(*derivs)(x+h,yt,dyt);
	
	
	
	for (i=1;i<=n;i++)
		yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
	free_dvector(yt,1,n);
	free_dvector(dyt,1,n);
	free_dvector(dym,1,n);
}


