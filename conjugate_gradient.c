#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SIGMA 0.4f
#define RHO 0.5f
#define TOL 1.0e-10f


/* Returns rosenbrock function to minimize */
float rosenbrock(float *x) {
    return 100.0f*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1.0f-x[0])*(1.0f-x[0]);
}


/* Return the gradient of the function */
void rosenbrock_grad(float *x, float *grad) {
    grad[0] = -400.0f*(x[1]-x[0]*x[0])*x[0] - 2.0f * (1.0f-x[0]);
    grad[1] = 200.0f*(x[1]-x[0]*x[0]);
}


/* P_k for the algorithm */
void pk_func(float *gradient, float beta, float *pk) {
    for(int i=0; i<2; i++)
        pk[i] = -gradient[i] + beta * pk[i];
}


/* Backtrackr algorithm to perform a line search,
   as found in Quarteroni's book */
void backtrackr(
        float sigma, // (0,1/2)
        float rho,   // (0,1)
        float *x,
        float(f)(float *), // function
        void(J)(float *,float *), // Jacobian (gradient)
        float *pk
        ) {
    float alpha, fk, fk1, *Jfk, *xx;
    Jfk = (float*) malloc( 2 * sizeof(float));
    xx = (float*) malloc( 2 * sizeof(float));
    alpha = 1.0f;
    fk = f(x);
    J(x, Jfk);
    for(int i=0; i<2; i++) {
        xx[i] = x[i];
        x[i] = x[i] + alpha * pk[i];
    }

    fk1 = f(x) - sigma * alpha * (Jfk[0]*pk[0]+Jfk[1]*pk[1]);

    while (fk1 > fk) {
        alpha = alpha*rho;
        for(int i=0; i<2; i++)
            x[i] = xx[i] + alpha * pk[i];
        fk1 = f(x) - sigma * alpha * (Jfk[0]*pk[0]+Jfk[1]*pk[1]);
    }
}


float beta_func(float *grad, float *grad_old) {
    float beta;
    // Ratio of dot products (Fletcher-Reeves)
    // beta = (grad[0]*grad[0]+grad[1]*grad[1]);

    // Polak-Ribière
    beta = grad[0] * (grad[0] - grad_old[0]) + grad[1] * (grad[1] - grad_old[1]);

    beta = beta /(grad_old[0]*grad_old[0]+grad_old[1]*grad_old[1]);

    return beta;
}


int main (int argc, char *argv[]) {
    float *x, *grad, *grad_old, *pk;
    float function, beta;
    int iter, iter_max=1000;

    x        = (float*) malloc( 2 * sizeof(float));
    grad     = (float*) malloc( 2 * sizeof(float));
    grad_old = (float*) malloc( 2 * sizeof(float));
    pk       = (float*) malloc( 2 * sizeof(float));

    // Default initial conditions, can be changed via argv
    x[0] = -1.5f;
    x[1] = -1.0f;
    if (argc>1) {x[0] = atof(argv[1]);}
    if (argc>2) {x[1] = atof(argv[2]);}

    function = rosenbrock(x);
    rosenbrock_grad(x, grad);
    for(int i=0; i<2; i++)
        pk[i] = -grad[i];
    printf("function: %f, position: (%f,%f)\n", function, x[0], x[1]);

    iter = 1;
    while (function > TOL && iter < iter_max) {
        iter++;
        backtrackr(SIGMA, RHO, x, rosenbrock, rosenbrock_grad, pk);
        for(int i=0; i<2; i++)
            grad_old[i] = grad[i];
        rosenbrock_grad(x, grad);
        beta = beta_func(grad, grad_old);
        pk_func(grad, beta, pk);

        function = rosenbrock(x);
        printf("function: %f, position: (%f,%f)\n", function, x[0], x[1]);
    }
    printf("function minumum: %f, found after %d iterations at position: (%f,%f)\n",
            function, iter, x[0], x[1]);
}
