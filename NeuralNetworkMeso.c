#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nrutil.h"


FILE  *Averages_NeuralNetwork, *Spikes;


void rk4(double y[], double dydx[], int n, double x, double h, double yout[],double eta, double I_B, double I_S, double tau_e, double N, double tau_d, double tau_f, double U_0,
	void (*derivs)(double, double [], double [], double, double, double, double, double, double, double, double));

void derivs(double x,double y[],double dydx[], double eta, double I_B, double I_S, double tau_e, double N, double tau_d, double tau_f, double U_0);
void derivs2(double x,double y[],double dydx[], double eta, double I_B, double I_S, double tau_e, double N, double tau_d, double tau_f, double U_0);

double *f,*g, *df,*dg, *V, *etas;



double random_lorentzian(double x0, double gamma) {

    double u0 = (double)rand() / RAND_MAX;
 
    return x0 + gamma * tan(3.14159265358979323846 * (u0 - 0.5));
}


// neural network QIF
int main(){

double t;
double S;
double rtot;
const double maxt=1000. ;
const double h=0.0015 ; 
const double N = 2000 ; //  recordar poner 200.000
const double H = 0.0;
const double delta = 0.25;
const double I_B = -1.0;
const double J = 15;
const double tau_e = 15;

const double PI = 3.14159265358979323846;
const double Vp = 100; 
const double Vr = -100;
double U_0 = 0.2; 
double x_0 = 0.73;
double u_0 = 0.59;
const double tau_d = 200;
const double tau_f = 1500;
double I_S;
double x = x_0;
double u = u_0;

f=dvector(1,1);
df=dvector(1,1);
g=dvector(1,2);
dg=dvector(1,2);

V = dvector(1,N+1);
etas = dvector(1, N + 1);



Averages_NeuralNetwork=fopen("averages_meso.dat","w");
Spikes=fopen("spikes_meso.dat","w");




// initial condition

for (int i = 1; i <= N; i++) {
        etas[i] = random_lorentzian(0, 0.25);
        V[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
    }  
f[1] = -100.0;
g[1] = x_0;
g[2] = u_0;
t=0.0; // tiempo inicial

	while (t<=maxt){
        S = 0.0;
        
        if ((t >=150 && t<= 300) || (t >=450 && t<= 600)){
            I_S = 2;
        }else{
            I_S = 0;
        }

        for (int i = 1; i <= N; i++) { // Neuronas
        
            /* Numeric scheme */
            f[1] = V[i];
           
            double eta = etas[i];
        
          
            derivs(t,f,df,eta, I_B, I_S, tau_e, N, tau_d, tau_f, U_0);
            rk4(f,df,1,t,h,f,eta, I_B, I_S, tau_e, N, tau_d, tau_f, U_0,derivs);
            
            V[i] = f[1];
        

            if (V[i]>= Vp){
                fprintf(Spikes, "%lf %d\n", t, i);
                S+=1;
                V[i] = Vr;
               
              
            }
             

        }
        derivs2(t,g,dg,0, I_B, I_S, tau_e, N, tau_d, tau_f, U_0);
        rk4(g,dg,2,t,h,f,0, I_B, I_S, tau_e, N, tau_d, tau_f, U_0,derivs2);

        x = g[1];
        u = g[2];
        x = x - u*x*S/N;
        u = u + U_0*(1-u)*S/N;
        for (int i = 1; i <= N; i++) { // Neuronas
            V[i] = V[i]+J*x*u*S/N;
            
        }


        fprintf(Averages_NeuralNetwork, "%lf %lf %lf\n", x, u, t); 

        t=t+h;

	}

fclose(Averages_NeuralNetwork);
fclose(Spikes);

free_dvector(f,1,1);
free_dvector(df,1,1);
free_dvector(g,1,2);
free_dvector(dg,1,2);
free_dvector(etas, 1, N + 1);
free_dvector(V, 1, N + 1);


return 0;
}





void derivs(double x,double y[],double dydx[],double eta, double I_B, double I_S, double tau_e, double N, double tau_d, double tau_f, double U_0)
{
    dydx[1] = (y[1]*y[1] + eta + I_B + I_S)/tau_e;
 
}

void derivs2(double x,double y[],double dydx[],double eta, double I_B, double I_S, double tau_e, double N, double tau_d, double tau_f, double U_0)
{
    dydx[1] = (1-y[1])/tau_d;
    dydx[2] = (U_0-y[2])/tau_f;
}





void rk4(double y[], double dydx[], int n, double x, double h, double yout[],double eta, double I_B, double I_S, double tau_e, double N, double tau_d, double tau_f, double U_0,
	void (*derivs)(double, double [], double [],double, double, double, double, double, double, double, double))
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
	(*derivs)(xh,yt,dyt,eta,I_B,I_S,  tau_e,  N,  tau_d,  tau_f,  U_0);
	for (i=1;i<=n;i++) yt[i]=y[i]+hh*dyt[i];
	(*derivs)(xh,yt,dym, eta,I_B,I_S,  tau_e,  N,  tau_d,  tau_f,  U_0);
	for (i=1;i<=n;i++) {
		yt[i]=y[i]+h*dym[i];
		dym[i] += dyt[i];
	/*fprintf(fdue,"%d%.10e %.10e\n",i,dym[i],y[i]);*/
	}
	(*derivs)(x+h,yt,dyt,eta,I_B,I_S,  tau_e,  N,  tau_d,  tau_f,  U_0);
	
	
	
	for (i=1;i<=n;i++)
		yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
	free_dvector(yt,1,n);
	free_dvector(dyt,1,n);
	free_dvector(dym,1,n);
}


