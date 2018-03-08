#include <stdio.h>
#include <stdlib.h>
// #include <stdarg.h>
#include <stdint.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fftw3.h>
#include "cblas.h"
#include <lapacke.h>
#include "matrix_utils.h"


#include "spidriver_host.h"
#include "adcdriver_host.h"

// Data matrix size
#define NUM_FRAMES 100
#define NUM_PTS 1024

// Margin for anomaly detection threshold
#define MARGIN 2.8f

#define ZEROTHRESH 1.0e-7

// Statically allocate stuff having to do with spectra and matrices.
float volts[NUM_PTS];          // Vector of measurements
float M0[NUM_PTS*NUM_FRAMES];  // NUM_PTS rows,  NUM_FRAMES cols
float M[NUM_PTS*NUM_FRAMES];   // Copy of M0 which gets destroyed in svd call
float x[NUM_PTS];
float y[NUM_PTS];
fftwf_complex yspec[NUM_PTS];
fftwf_plan p;
float pow_learn_thresh;

// SVD stuff
// I will use column major -- all rows of one column comes before the next column.
// Each column will be a spectrum stored as a column vector. 
#define MAX_RANK 20
float U[NUM_PTS * NUM_PTS];
float S[NUM_FRAMES];           // S is a vector
float VT[NUM_FRAMES * NUM_FRAMES];
float superb[NUM_PTS-1];
float Mr[NUM_PTS*MAX_RANK];    // Reduced U matrix.
float T1[NUM_PTS*NUM_PTS];     // Temp matrix
float T2[NUM_PTS*NUM_PTS];     // Temp matrix
float Nr[NUM_PTS*NUM_PTS];     // Null space projection matrix.
float Id[NUM_PTS*NUM_PTS];     // Identity matrix
int P[NUM_PTS*NUM_PTS];        // Permutation matrix.

//===========================================================
// Utility functions


//-----------------------------------------------------
// Move this into matrix_utils
void extract_cols(float *A, int m, int n, int c, float *E) {
  // This fcn takes input matrix A of size mxn.  It extracts the first
  // c col vectors of A, from 0 ... c-1.  It then puts the extracted
  // vectors into E.  E has size [m, c]

  int i, j;
  for (i = 0; i < c; i++) {      // Loop over cols
    for (j = 0; j < m; j++) {    // Loop over rows
      E[i*m+j] = A[i*m+j];
    }
  }
}


//-----------------------------------------------------
// Called when Ctrl+C is pressed - triggers the program to stop.
void stopHandler(int sig) {
  adc_quit();
  exit(0);
}


//==========================================================
// This is main program which exercises the A/D.  It opens a file
// and dumps a dataframe into the file.  Then it waits for a period
// then does it again.
int main (void)
{
  // Loop variables
  uint32_t i;
  uint32_t j;

  // Buffers for tx and rx data from A/D registers.
  uint32_t tx_buf[3];
  uint32_t rx_buf[4];

  // Numerical helpers
  float vmean;
  float re, im;
  float er;      // Effective matrix rank
  float nnr;     // Normalized norm of vector in nullspace.
  float thresh;  // Anomaly threshold.
  float tmp1, tmp2;
  float pr;  // Used in computing effective rank of U
  float H;   // Used in computing effective rank of U
  int r;     // Used to extract cols from U

  // Stuff used with LAPACK calls
  int info;
  // I need to make these real variables so I can pass them to LAPACK.
  int m = NUM_PTS;     // Num rows
  int n = NUM_FRAMES;  // Num cols
  int lda = m;
  int ldu = m;
  int ldv = n;

  // File ID
  FILE *fp;

  // filename
  char filename[20];

  // Stuff used with "hit return when ready..." 
  char dummy[8];

  printf("------------   Starting main.....   -------------\n");

  // Run until Ctrl+C pressed:
  signal(SIGINT, stopHandler);

  // Sanity check user.
  if(getuid()!=0){
     printf("You must run this program as root. Exiting.\n");
     exit(EXIT_FAILURE);
  }

  // Initialize A/D converter
  adc_config();

  // Zero out filename string
  memset(filename, 0x00, 20);


  // Test PRU RAM
  printf("--------------------------------------------------\n");
  i = 123456;
  printf("About to test PRU RAM.  Write value = %d\n", i);
  j = pru_test_ram(1, i);
  printf("Just wrote value to PRU RAM and then read it back.  Value read = %d\n", j);


  // Test PRU communication link
  printf("--------------------------------------------------\n");
  printf("About to test PRU communication link\n");
  j = pru_test_communication();
  printf("Just did communication test.  Number of cycles for read = %d\n", j);


  // Now check the A/D is alive by reading from its config reg.
  printf("--------------------------------------------------\n");
  printf("About to read A/D config register\n");
  rx_buf[0] = adc_get_id_reg();
  printf("Read ID reg.  Received ID = 0x%08x\n", rx_buf[0]);


  // Set sample rate to 32kSPS, chan 0.
  printf("--------------------------------------------------\n");
  printf("Set sample rate to 32kSPS and set channel 0\n");
  adc_set_samplerate(SAMP_RATE_31250);
  adc_set_chan0();


  // Set up FFT stuff
  p = fftwf_plan_dft_r2c_1d(NUM_PTS, y, yspec, FFTW_ESTIMATE);


  //==================================================================
  //  Now do the main business of this program.
  printf("--------------------------------------------------\n");
  printf("Now read training set: %d data frames\n", NUM_FRAMES);
  printf("Hit return when ready -->\n");
  fgets (dummy, 8, stdin);

  // ---------------------------------------------
  // Read A/D, take FFT, then fill out M0 matrix with spectra
  for (i=0; i<NUM_FRAMES; i++) {

    // Read in data frame and store in a vector.
    adc_read_multiple(NUM_PTS, volts);

    // Check for saturation and compute mean.  Max input voltage is about 4V.
    vmean = 0.0f;
    for (j=0; j<NUM_PTS; j++) {
      if (fabs(volts[j]) > 3.999) {
         printf("!!!!!!  Input saturated  !!!!!!\n");
      }
      vmean += volts[j];
    }
    vmean = vmean/((float) NUM_PTS);

    // subtract mean
    for (j=0; j<NUM_PTS; j++) {
      y[j] = volts[j] - vmean;
    }

    // Do FFT.  The input is y, the output is yspec
    fftwf_execute(p);

    // Normalize and threshold spectrum.
    // First find real & imag parts and send result to y[j]
    tmp1 = 0;
    for (j=0; j<NUM_PTS; j++) {
      re = yspec[j][0];
      im = yspec[j][1];
      y[j] = re*re + im*im;  // Power spectral density
      tmp1 = tmp1 + y[j];
      // printf("y[%d] = %f, tmp1 = %e\n", j, y[j], tmp1);
    }

    // Total power of input power spectrum.
    // This is threshold -- if input spectrum falls
    // below this value, then ignore input (don't 
    // look for anomalies).
    pow_learn_thresh = tmp1/1000.0f;

    for (j=0; j<NUM_PTS; j++) {
      y[j] = y[j]/tmp1;      // Normalize
      if (y[j] < ZEROTHRESH) {
        y[j] = 0.0f;
      }
    }

    // Copy into M0
    for (j=0; j<NUM_PTS; j++) {
      M0[i*NUM_PTS + j] = y[j];
      // printf("M0[%d][%d] = %f\n", j, i, M0[i*NUM_PTS+j]); 
    }

    usleep(30000);   // delay 30ms sec before reading next frame
  }

//  printf("M0 (linear) = \n");
//  print_matrix_linear(M0, m, n);

//  printf("M0 = \n");
//  print_matrix(M0, m, n);


  // ---------------------------------------------
  // Now take SVD(M0) and extract the columns of U0 for the low-rank approx
  // Note that the input matrix is destroyed, so I need to pass in a copy.
  for (j=0; j<NUM_PTS*NUM_FRAMES; j++) {
    M[j] = M0[j];
  }
  // Must make sure I call this correctly for rectangular matrix.
  // m = NUMROWS, n = NUMCOLS, lda = m, ldu = m, ldv = n 
  info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'A', 'A', 
                  m, n, M, 
                  lda, S, U, ldu, 
                  VT, ldv, superb);
  if (info != 0)  {
    fprintf(stderr, "Error: sgesvd returned with a non-zero status (info = %d)\n", info);
    return(-1);
  }

  //printf("S = \n");
  //print_matrix(S, n, 1);

  //printf("U = \n");
  //print_matrix(U, m, m);

  // Get effective rank of M0 to know how many vectors to pull out.
  pr = 0.0f;
  for (j=0; j<NUM_FRAMES; j++) {
    pr = pr+fabs(S[j]);
  }
  for (j=0; j<NUM_FRAMES; j++) {
    x[j] = S[j]/pr;
  }
  H = 0.0f;
  for (j=0; j<NUM_FRAMES; j++) {
    if (x[j] > 0) {
      H = H+x[j]*log(x[j]);
    }
  }
  r = floor(exp(-H));
  // Guard against problems.
  if (r > MAX_RANK) {
    r = MAX_RANK;
  }
  // r = r/2.0f;

  printf("r = %d\n", r);

  // Pull out r cols
  extract_cols(U, m, m, r, Mr);

  //printf("Mr = \n");
  //print_matrix(Mr, m, r);

  // Now create projection operator into null space
  // Projection operator is I - Mr*((Mr'*Mr)\Mr');
  // This is multi-step process in C.
  // T1 = Mr'*Mr.  Return is size COLSxCOLS
  // Mr = NUM_PTS x r
  // T1 = r x r
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                r, r, m, 1.0f, Mr, m, Mr, m, 0.0f, T1, r);

  //printf("After sgemm 2.  This should be identity -- T1 = Mr'*Mr = \n");
  //print_matrix(T1, r, r);

  // T1 = inv(Mr'*Mr).  
  // T1 = r x r
  info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, r, r, T1, r, P);
  if (info != 0)  {
    fprintf(stderr, "Error: sgetrf returned with a non-zero status (info = %d)\n", info);
    return(-1);
  }

  //printf("This should be identity -- After sgetrf, T1 = \n");
  //print_matrix(T1, r, r);

  // sgetri overwrites T1 with the output.
  info = LAPACKE_sgetri (LAPACK_COL_MAJOR, r, T1, r, P);
  if (info != 0)  {
    fprintf(stderr, "Error: sgetri returned with a non-zero status (info = %d)\n", info);
    return(-1);
  }

  //printf("This should be identity -- After sgetri, T1 = \n");
  //print_matrix(T1, r, r);

  // T2 = T1*Mr'
  // T1 = r x r
  // Mr = NUM_PTS x r (before transpose)
  // T2 = r x NUM_PTS
  cblas_sgemm(CblasColMajor, 
		CblasNoTrans, CblasTrans,
                r, m, r, 
		1.0f, T1, r, 
		Mr, m, 
		0.0f, T2, r);

  //printf("T2 = T1*Mr'\n");
  //print_matrix(T2, r, m);

  // T1 = Mr*T2
  // T1 = NUM_PTS x NUM_PTS
  cblas_sgemm(CblasColMajor, 
		CblasNoTrans, CblasNoTrans,
                m, m, r, 
		1.0f, Mr, m, 
		T2, r, 
		0.0f, T1, m);

  //printf("T1 = Mr*T2 = \n");
  //print_matrix(T1, m, m);

  // Nr = I - T1
  eye(m, m, Id);
  //printf("Id = \n");
  //print_matrix(Id, m, m);

  for (i=0; i<NUM_PTS*NUM_PTS; i++) {
    Nr[i] = Id[i] - T1[i];
  }

  //printf("Nr = \n");
  //print_matrix(Nr, m, m);


  // ---------------------------------------------
  // Now that we have the projection matrix Nr, compute
  // good threshold for anomalies using a new set of data.
  thresh = 0.0f;
  for (i=0; i<NUM_FRAMES; i++) {
    // Read in data frame and store in a vector.
    adc_read_multiple(NUM_PTS, volts);

    // compute mean
    vmean = 0.0f;
    for (j=0; j<NUM_PTS; j++) {
      if (fabs(volts[j]) > 3.999) {
         printf("!!!!!!  Input saturated  !!!!!!\n");
      }
      vmean += volts[j];
    }
    vmean = vmean/((float) NUM_PTS);

    // subtract mean
    for (j=0; j<NUM_PTS; j++) {
      y[j] = volts[j] - vmean;
    }

    // Do FFT.  The input is y, the output is yspec
    fftwf_execute(p);

    // Normalize and threshold spectrum.
    // First find real & imag parts and send result to x[j]
    tmp1 = 0;
    for (j=0; j<NUM_PTS; j++) {
      re = yspec[j][0];
      im = yspec[j][1];
      x[j] = re*re + im*im;  // Power spectral density
      tmp1 = tmp1 + x[j];
    }

    for (j=0; j<NUM_PTS; j++) {
      x[j] = x[j]/tmp1;      // Normalize
      if (x[j] < ZEROTHRESH) {
        x[j] = 0.0f;
      }
    }

    //printf("x = \n");
    //print_matrix(x, m, 1);

    // get norm of Nr*x.
    // y = Nr*x
    cblas_sgemv(CblasColMajor, CblasNoTrans, m, m, 1.0f, Nr, m, x, 1, 0.0f, y, 1);

    //printf("y = \n");
    //print_matrix_linear(y, m, 1);

    tmp1 = LAPACKE_slange (CblasColMajor, '1', m, 1, y, m);
    //printf("norm(Nr*x) = %f\n", tmp1);
    
    // Get norm of x
    tmp2 = LAPACKE_slange (CblasColMajor, '1', m, 1, x, m);
    //printf("norm(x) = %f\n", tmp2);
    nnr = tmp1/tmp2;
    printf("nnr = %f\n", nnr);
    //if (nnr > thresh) {
    //  thresh = nnr;
    //}
    thresh += nnr;
  }
  thresh = thresh/((float) NUM_FRAMES);

  // Increase threshold by some margin
  thresh = MARGIN*thresh;
  printf("thresh = %f\n", thresh);

  //===================================================================
  // ---------------------------------------------
  // Now that we have the projection matrix Nr, go into
  // loop.  Take data frame, use Nr to project into nullspace
  // of M0, and check size of signal in nullspace. 

  printf("--------------------------------------------------\n");
  printf("Ready to monitor for anomalies.\n");
  printf("Hit return when ready -->\n");
  fgets (dummy, 8, stdin);

  while(1) {
    printf("------------\n");

    // Read in data frame.
    adc_read_multiple(NUM_PTS, volts);

    // compute mean
    vmean = 0.0f;
    for (j=0; j<NUM_PTS; j++) {
      if (fabs(volts[j]) > 3.999) {
         printf("!!!!!!  Input saturated  !!!!!!\n");
      }
      vmean += volts[j];
    }
    vmean = vmean/((float) NUM_PTS);

    // subtract mean
    for (j=0; j<NUM_PTS; j++) {
      y[j] = volts[j] - vmean;
    }

    //printf("y = \n");
    //print_matrix(y, m, 1);

    // Do FFT.  The input is y, the output is yspec
    fftwf_execute(p);

    // Normalize and threshold spectrum.
    // First find real & imag parts and send result to x[j]
    tmp1 = 0;
    for (j=0; j<NUM_PTS; j++) {
      re = yspec[j][0];
      im = yspec[j][1];
      x[j] = re*re + im*im;  // Power
      tmp1 = tmp1 + x[j];
    }

    // If spectral power is < 1/10th power of learning signal,
    // then just continue -- don't want to look at noise
    // without a signal.
    if (tmp1 < pow_learn_thresh) {
      printf("Power too small....\n");
      usleep(50000);
      continue;
    }

    for (j=0; j<NUM_PTS; j++) {
      x[j] = x[j]/tmp1;      // Normalize
      if (x[j] < ZEROTHRESH) {
        x[j] = 0.0f;
      }
    }

    // Project onto nullspace
    // nnr = norm(Nr*yspec)/norm(yspec);

    //printf("x = \n");
    //print_matrix(x, m, 1);

    // get norm of Nr*x.
    // y = Nr*x
    cblas_sgemv(CblasColMajor, CblasNoTrans, m, m, 1.0f, Nr, m, x, 1, 0.0f, y, 1);

    //printf("y = Nr*x = \n");
    //print_matrix(y, m, 1);

    tmp1 = LAPACKE_slange (CblasColMajor, '1', m, 1, y, m);
    //printf("norm(Nr*x) = %f\n", tmp1);

    // Get norm of x
    tmp2 = LAPACKE_slange (CblasColMajor, '1', m, 1, x, m);
    //printf("norm(x) = %f\n", tmp2);
    nnr = tmp1/tmp2;
    printf("thresh = %f,  nnr = %f", thresh, nnr);

    // Test for anomaly
    if (nnr > thresh) {
      printf("  ...  Anomaly detected!!!\n");
    } else {
      printf("\n");
    }

    //printf("Hit return to continue -->\n");
    //fgets (dummy, 8, stdin);
    usleep(100000);

  }  // while(1)

}

