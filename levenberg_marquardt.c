#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOL 1.0e-10f
#define TOL2 1.0e-6f
#define TAU 1.0e-2f

/* Returns function F to minimize, half of rosenbrock function */
float rosenbrock_F(float *x) {
    return 50.0f*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + 0.50f*(1.0f-x[0])*(1.0f-x[0]);
}

/* Fills f vector, decomposition of F into squared functions */
void rosenbrock_f(float *x, float *f) {
    f[0] = 10.0f*(x[1]-x[0]*x[0]);
    f[1] = (1.0f-x[0]);
}


// Fill the Jacobian of the function f
void rosenbrock_f_jacobian(float *x, float *jacobian) {
    jacobian[0] = -20.0f * x[0];
    jacobian[1] = 10.0f;
    jacobian[2] = -1.0f;
    jacobian[3] = 0.0f;
}

/* compute hessian approximation as Jacobian^T * Jacobian */
void hessian_approx_func(float *jacobian, float *hessian_approx) {
    hessian_approx[0] = jacobian[0]*jacobian[0] + jacobian[2]*jacobian[2];
    hessian_approx[1] = jacobian[0]*jacobian[1] + jacobian[2]*jacobian[3];
    hessian_approx[2] = hessian_approx[1];
    hessian_approx[3] = jacobian[1]*jacobian[1] + jacobian[3]*jacobian[3];
}

/* Returns inverted matrix in the memory address as the given matrix */
void invert_matrix(float *matrix) {
    float det = fabs(matrix[0]*matrix[3]-matrix[1]*matrix[2]);
    float temp = matrix[0];
    matrix[0] = matrix[3] / det;
    matrix[3] = temp / det;
    matrix[1] = -matrix[1] / det;
    matrix[2] = -matrix[2] / det;
}

/* Levenerg-Marquardt Algorithm, following pseudocode from
   H. B. Nielsen K. Madsen and O. Tingle.Methods for Non-Linear Least Squares Problems
   returns number of iterations */
int algorithm(float *x) {
    float *A, *x_new, *J, *f, *g, *h_lm;
    float x_norm2;
    float rho;
    float g_inf, h_norm2;
    float mu, nu=2.0f;
    int k=0, k_max=10000;

    x_new = (float *) malloc( 2 * sizeof(float ));
    A     = (float *) malloc( 4 * sizeof(float ));
    J     = (float *) malloc( 4 * sizeof(float ));
    f     = (float *) malloc( 2 * sizeof(float ));
    g     = (float *) malloc( 2 * sizeof(float ));
    h_lm  = (float *) malloc( 2 * sizeof(float ));

    rosenbrock_f(x, f);
    rosenbrock_f_jacobian(x, J);
    hessian_approx_func(J, A);
    g[0] = J[0] * f[0] + J[2] * f[1];
    g[1] = J[1] * f[0] + J[3] * f[1];

    g_inf = fmax( fabs(g[0]), fabs(g[1]) );
    mu = TAU * fmax( fmax(A[0],A[1]), fmax(A[2],A[3]) );

    while (g_inf > TOL && k < k_max) {
        A[0] += mu;
        A[3] += mu;
        invert_matrix(A);
        h_lm[0] = -A[0]*g[0] - A[1]*g[1];
        h_lm[1] = -A[2]*g[0] - A[3]*g[1];

        x_norm2 = sqrt(x[0]*x[0] + x[1]*x[1]);
        h_norm2 = sqrt(h_lm[0]*h_lm[0] + h_lm[1]*h_lm[1]);

        if (h_norm2 < TOL2*(x_norm2 + TOL2)) {
            printf("TOL2 break\n");
            break;
        }
        else{
            for(int i=0; i<2; i++)
                x_new[i] = x[i] + h_lm[i];

            rho = 0.5f*( h_lm[0]*(mu*h_lm[0]-g[0]) + h_lm[1]*(mu*h_lm[1]-g[1]) );
            rho = ( rosenbrock_F(x) - rosenbrock_F(x_new) )/rho;
            if (rho > 0) {
                k += 1;
                printf("function: %f, position: (%f,%f)\n", 2.0f*rosenbrock_F(x_new), x_new[0], x_new[1]);
                for(int i=0; i<2; i++)
                    x[i] = x_new[i];

                rosenbrock_f(x, f);
                rosenbrock_f_jacobian(x, J);
                hessian_approx_func(J, A);
                g[0] = J[0] * f[0] + J[2] * f[1];
                g[1] = J[1] * f[0] + J[3] * f[1];

                g_inf = fmax( fabs(g[0]), fabs(g[1]) );
                mu = mu * fmax( 0.33f, 1.0f-pow(2.0f*rho-1.0f,3) );
                nu = 2.0f;
            }
            else{
                mu = mu * nu;
                nu = 2.0f * nu;
            }
        }
    }
    return k;
}

int main (int argc, char *argv[]) {
    float *x;
    int k;

    x = (float*) malloc( 2 * sizeof(float));

    // Default initial conditions, can be changed via argv
    x[0] = -1.5f;
    x[1] = -1.0f;
    if (argc>1) {x[0] = atof(argv[1]);}
    if (argc>2) {x[1] = atof(argv[2]);}

    printf("function: %f, position: (%f,%f)\n", 2.0f*rosenbrock_F(x), x[0], x[1]);

    /* call algorithm */
    k = algorithm(x);

    printf("function minumum: %f, found after %d iterations at position: (%f,%f)\n",
            2.0f*rosenbrock_F(x), k, x[0], x[1]);
}
