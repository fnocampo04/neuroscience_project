#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


FILE  *rA_montbrio, *rB_montbrio, *rtot_montbrio,  *vA_montbrio, *vB_montbrio /*, *sA_montbrio, *sB_montbrio*/;


void rk4(double y[], double dydx[], int n, double x, double h, double yout[],
	void (*derivs)(double, double [], double []));

void derivs(double x,double y[],double dydx[]);
//void derivs2(double x,double y[],double dydx[]);


double *f, *df;
const double JAA= -10;
const double JBB= -20;
const double JBA= -8;
const double JAB= -0;
const double deltaA= 0.01 ;
const double deltaB= 0.01 ;
const double etamedioA= 1. ;
const double etamedioB= 1. ;
const double tau=1. ;
const double tau_dA= 0.25 ;
const double tau_dB= 8. ;



 
//sistema 6X6 di montbrio per 2 famiglie integrata con Runge-Kutta 4ordine
int main(){

double t;
double rtot;
const double maxt=10. ;
const double h=0.01 ; 


f=dvector(1,6);
df=dvector(1,6);

rA_montbrio=fopen("rA.dat","w");
vA_montbrio=fopen("vA.dat","w");
//sA_montbrio=fopen("sA.dat","w");
rB_montbrio=fopen("rB.dat","w");
vB_montbrio=fopen("vB.dat","w");
//sB_montbrio=fopen("sB.dat","w");
rtot_montbrio=fopen("rtot.dat","w");


    // initial condition
        f[1]= 11.38;//rA(t)
        f[2]= 11.425;//rB(t)
	f[3]= 0.;//vA(t)
        f[4]= 0.;//vB(t)
        f[5]= 0.;//sA(t)
	f[6]= 0.;//sB(t)
        
        t=0.015; 
	while (t<=maxt){

		/* Numeric scheme */
               	derivs(t,f,df);
                rk4(f,df,6,t,h,f,derivs);

   		t=t+h; 
                                     
                fprintf(rA_montbrio,"%lf %lf \n", t, f[1]); // rA(t) montbrio dynamics plot
                fprintf(rB_montbrio,"%lf %lf \n", t, f[2]); // rB(t) montbrio dynamics plot
                fprintf(vA_montbrio,"%lf %lf \n", t, f[3]); // vA(t) montbrio dynamics plot
                fprintf(vB_montbrio,"%lf %lf \n", t, f[4]); // vB(t) montbrio dynamics plot
      //          fprintf(sA_montbrio,"%lf %lf \n", t, f[5]); // sA(t) montbrio dynamics plot
      //          fprintf(sB_montbrio,"%lf %lf \n", t, f[6]); // sB(t) montbrio dynamics plot

		rtot=(f[1]+f[2])/2;
		fprintf(rtot_montbrio,"%lf %lf \n", t, rtot); // rtot(t) montbrio dynamics plot

	}

fclose(rA_montbrio);
fclose(vA_montbrio);
//fclose(sA_montbrio);
fclose(rB_montbrio);
fclose(vB_montbrio);
//fclose(sB_montbrio);
fclose(rtot_montbrio);


free_dvector(f,1,6);
free_dvector(df,1,6);

return 0;
}





void derivs(double x,double y[],double dydx[])
{
 
 dydx[1]=deltaA/M_PI +2*y[1]*y[3];
 dydx[3]=y[3]*y[3]+1.-pow(M_PI*y[1],2)+JAA*y[5]+JBA*y[6];
 dydx[5]=(y[1]-y[5])/tau_dA;

 dydx[2]=deltaB/M_PI +2*y[2]*y[4];
 dydx[4]=y[4]*y[4]+1.-pow(M_PI*y[2],2)+JBB*y[6]+JAB*y[5];
 dydx[6]=(y[2]-y[6])/tau_dB;
 
 
 
 /*dydx[1]= (mu[0]*h[0]+ mu[1]*h[1])*(1.-y[1]*y[1])*cos(y[1+dim]+beta)/2.;
 dydx[2]= (mu[0]*h[1]+ mu[1]*h[0])*(1.-y[2]*y[2])*cos(y[2+dim]+beta)/2.;
 
 dydx[3]= 1. - (mu[0]*h[0]+  mu[1]*h[1])*(1.+y[1]*y[1])*sin(y[1+dim]+beta)/(2.*y[1]);
 dydx[4]= 1. - (mu[0]*h[1]+  mu[1]*h[0])*(1.+y[2]*y[2])*sin(y[2+dim]+beta)/(2.*y[2]);*/
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







