#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nrutil.h"

FILE *Averages_NeuralNetwork, *spikes_1, *spikes_2;

void rk4(double y[], double dydx[], int n, double x, double h, double yout[], double eta_1, double eta_2, double eta_3, double I_B, double I_S_1, double I_S_2, double tau_e, double N, double tau_d, double tau_f, double U_0,
    void (*derivs)(double, double [], double [], double, double, double, double, double, double, double, double, double, double, double));

void derivs(double x, double y[], double dydx[], double eta_1, double eta_2, double eta_3, double I_B, double I_S_1, double I_S_2, double tau_e, double N, double tau_d, double tau_f, double U_0);
void derivs2(double x, double y[], double dydx[], double eta_1, double eta_2, double eta_3, double I_B, double I_S_1, double I_S_2, double tau_e, double N, double tau_d, double tau_f, double U_0);

double *f, *g, *df, *dg, *V_1,*V_2,*V_0, *etas_1, *etas_2, *etas_0;
double A_1; // Variable global para almacenar A_1(t)
double A_2; // Variable global para almacenar A_2(t)
double A_0; // Variable global para almacenar A_0(t)

double random_lorentzian(double x0, double gamma) {
    double u0 = (double)rand() / RAND_MAX;
    return x0 + gamma * tan(3.14159265358979323846 * (u0 - 0.5));
}
// Para la columna B
int main() {
    double t;
    double S_1;
    double S_2;
    double S_0;
    const double maxt = 3000.;
    const double h = 0.0015;
    const double N = 2050; 
    double I_B = 1.532; // Columna B
    const double a = 0.4;
    const double Jee_c = 5*sqrt(a);
    const double Jee_s = 35*sqrt(a);
    const double Jie = 13*sqrt(a);
    const double Jei = -16*sqrt(a);
    const double Jii = -14*sqrt(a);
    const double tau_e = 15;
    const double Vp = 100;
    const double Vr = -100;
    double U_0 = 0.2;
    double x_0 = 0.52; // cerca de los valores estacionarios
    double u_0 = 0.48; 
    const double tau_d = 200;
    const double tau_f = 1500;
    const double H = 0.0;
    const double delta = 0.1;
    double I_S_1; // corriente para la excitatoria 1
    double I_S_2; // corriente para la excitatoria 2

    f = dvector(1, 3);
    df = dvector(1, 3);
    g = dvector(1, 4);
    dg = dvector(1, 4);

    V_1 = dvector(1, N + 1); // excitatoria 1
    V_2 = dvector(1, N + 1); // excitatoria 2
    V_0 = dvector(1, N + 1); // inhibidora

    etas_1 = dvector(1, N + 1);
    etas_2 = dvector(1, N + 1);
    etas_0 = dvector(1, N + 1);

    Averages_NeuralNetwork = fopen("averages_multipop1.dat", "w");
    spikes_1 = fopen("spikes_multipop1.dat", "w");
    spikes_2 = fopen("spikes_multipop2.dat", "w");

    // initial condition
    for (int i = 1; i <= N; i++) {
        etas_1[i] = random_lorentzian(H, delta);
        etas_2[i] = random_lorentzian(H, delta);
        etas_0[i] = random_lorentzian(H, delta);
        V_1[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        V_2[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        V_0[i] = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
    }
    g[1] = x_0; // para la excitatoria 1
    g[2] = u_0; // para la excitatoria 1

    g[3] = x_0; // para la excitatoria 2
    g[4] = u_0; // para la excitatoria 2

    t = -1000.0;

    while (t <= maxt) {
        S_1 = 0.0;
        S_2 = 0.0;
        S_0 = 0.0;
        I_S_1 = 0;
        I_S_2 = 0;
        I_B = 1.532; // Columna B

        if (t >= 0 && t <= 350) { // corriente para la excitatoria 1 COL B
            I_S_1 = 0.2;
        }

        if (t >2150) { 
            I_B = 1.2;
        }


        for (int i = 1; i <= N; i++) {
            /* Numeric scheme */
            f[1] = V_1[i];
            f[2] = V_2[i];
            f[3] = V_0[i];
            double eta_1 = etas_1[i];
            double eta_2 = etas_2[i];
            double eta_3 = etas_0[i];

            derivs(t, f, df, eta_1, eta_2, eta_3, I_B, I_S_1,I_S_2, tau_e, N, tau_d, tau_f, U_0);
            rk4(f, df, 3, t, h, f, eta_1, eta_2, eta_3, I_B, I_S_1,I_S_2, tau_e, N, tau_d, tau_f, U_0, derivs);
            V_1[i] = f[1];
            V_2[i] = f[2];
            V_0[i] = f[3];

            if (V_1[i] >= Vp) {
                fprintf(spikes_1, "%lf %d\n", t, i);
                S_1 += 1;
                V_1[i] = Vr;
            }if (V_2[i] >= Vp) {
                fprintf(spikes_2, "%lf %d\n", t, i);
                S_2 += 1;
                V_2[i] = Vr;
            }if (V_0[i] >= Vp) {
                S_0 += 1;
                V_0[i] = Vr;
            }
        }

        // A(t) = S / (N * h)
        A_1 = S_1 / (N * h);
        A_2 = S_2 / (N * h);
        A_0 = S_0 / (N * h);

        
        derivs2(t, g, dg, 0,0,0, I_B, I_S_1,I_S_2, tau_e, N, tau_d, tau_f, U_0);
        rk4(g, dg, 4, t, h, g, 0,0,0, I_B, I_S_1,I_S_2, tau_e, N, tau_d, tau_f, U_0, derivs2);

        double x_1 = g[1];
        double u_1 = g[2];
        double x_2 = g[3];
        double u_2 = g[4];


       
        for (int i = 1; i <= N; i++) {
            // Para V_1 (excitatoria 1)
            V_1[i] += (Jei * S_0 / N) + (Jee_s * x_1 * u_1 * S_1 / N) + (Jee_c * x_2 * u_2 * S_2 / N);

            // Para V_2 (excitatoria 2)
            V_2[i] += (Jei * S_0 / N)+(Jee_c * x_1 * u_1 * S_1 / N) + (Jee_s * x_2 * u_2 * S_2 / N) ;

            // Para V_0 (inhibidora)
            V_0[i] += (Jii * S_0 / N) + (Jie * S_1 / N) + (Jie * S_2 / N);
        }

        fprintf(Averages_NeuralNetwork, "%lf %lf %lf %lf %lf\n", x_1, u_1, x_2, u_2, t);

        t += h;
    }

    fclose(Averages_NeuralNetwork);
    fclose(spikes_1);
    fclose(spikes_2);

    free_dvector(f, 1, 3);
    free_dvector(df, 1, 3);
    free_dvector(g, 1, 4);
    free_dvector(dg, 1, 4);
    free_dvector(etas_1, 1, N + 1);
    free_dvector(etas_2, 1, N + 1);
    free_dvector(etas_0, 1, N + 1);
    free_dvector(V_1, 1, N + 1);
    free_dvector(V_2, 1, N + 1);
    free_dvector(V_0, 1, N + 1);


    return 0;
}

void derivs(double x, double y[], double dydx[], double eta_1, double eta_2, double eta_3, double I_B, double I_S_1, double I_S_2, double tau_e, double N, double tau_d, double tau_f, double U_0) {
    dydx[1] = (y[1] * y[1] + eta_1 + I_B + I_S_1) / tau_e; // excitatoria 1
    dydx[2] = (y[2] * y[2] + eta_2 + I_B + I_S_2) / tau_e; // excitatoria 2
    dydx[3] = (y[3] * y[3] + eta_3 + I_B) / tau_e; // inhibitoria
}

void derivs2(double x, double y[], double dydx[], double eta_1, double eta_2, double eta_3, double I_B, double I_S_1, double I_S_2, double tau_e, double N, double tau_d, double tau_f, double U_0) {
    (void)eta_1;(void)eta_2;(void)eta_3; (void)I_B; (void)I_S_1; (void)I_S_2; (void)tau_e; 

    dydx[1] = (1.0 - y[1]) / tau_d - y[2] * y[1] * A_1; // excitatoria 1
    dydx[2] = (U_0 - y[2]) / tau_f + U_0 * (1.0 - y[2]) * A_1;

    dydx[3] = (1.0 - y[3]) / tau_d - y[4] * y[3] * A_2; // excitatoria 2
    dydx[4] = (U_0 - y[4]) / tau_f + U_0 * (1.0 - y[4]) * A_2;



}


void rk4(double y[], double dydx[], int n, double x, double h, double yout[],double eta_1, double eta_2, double eta_3, double I_B, double I_S_1, double I_S_2, double tau_e, double N, double tau_d, double tau_f, double U_0,
	void (*derivs)(double, double [], double [],double, double,double,double,double, double, double, double, double, double, double))
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
	(*derivs)(xh,yt,dyt,eta_1, eta_2, eta_3,I_B,I_S_1,I_S_2,  tau_e,  N,  tau_d,  tau_f,  U_0);
	for (i=1;i<=n;i++) yt[i]=y[i]+hh*dyt[i];
	(*derivs)(xh,yt,dym, eta_1, eta_2, eta_3,I_B,I_S_1,I_S_2,  tau_e,  N,  tau_d,  tau_f,  U_0);
	for (i=1;i<=n;i++) {
		yt[i]=y[i]+h*dym[i];
		dym[i] += dyt[i];
	/*fprintf(fdue,"%d%.10e %.10e\n",i,dym[i],y[i]);*/
	}
	(*derivs)(x+h,yt,dyt,eta_1, eta_2, eta_3,I_B,I_S_1,I_S_2,  tau_e,  N,  tau_d,  tau_f,  U_0);
	
	
	
	for (i=1;i<=n;i++)
		yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
	free_dvector(yt,1,n);
	free_dvector(dyt,1,n);
	free_dvector(dym,1,n);
}


