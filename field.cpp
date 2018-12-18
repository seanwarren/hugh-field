#define __SSE2__ 1
#define __AVX__ 1

#define M_PI 3.14159265358979323846264338327950288

#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <random>
#include <functional>
#include <sleef.h>
#include <omp.h>

using namespace std;
using namespace std::literals::complex_literals;

typedef Sleef___m256d_2 m256d2;
typedef Sleef___m256_2 m256s2;

int main()
{
   int N = 1024 * 1024;
   int repeats = 100;

   vector<double> mask(N), phase(N);
   vector<complex<double>> field(N), field_ref(N);
   vector<float> mask_f(N), phase_f(N);
   vector<complex<float>> field_f(N);

   // Generate random numbers
   std::random_device rd;
   mt19937 gen(rd());
   uniform_real_distribution<double> phase_distribution(0, 2*M_PI);
   binomial_distribution<int> mask_distribution(1, 0.5);
   for (int i = 0; i < N; i++)
   {
      phase[i] = phase_distribution(gen);
      mask[i] = (double) mask_distribution(gen);

      phase_f[i] = (float) phase[i];
      mask_f[i] = (float) mask[i];
   }

   auto test_fcn = [&](const std::string name, function<void()> fcn)
   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<repeats; i++)
         fcn();
      auto diff = chrono::steady_clock::now() - start;
      cout << name << ": " << chrono::duration <double> (diff).count()  << " s" << endl;
   };


   // Reference zero order calculation
   test_fcn("baseline", [&](){ 

      for (int i = 0; i<N; i++)
         field_ref[i] = mask[i] * exp(1i * phase[i]);
      
   });

   // Parallel version of baseline
   test_fcn("parallel baseline", [&](){ 
      
      #pragma omp parallel for
      for (int i = 0; i<N; i++)
         field[i] = mask[i] * exp(1i * phase[i]);

   });

   // double AVX2 using SLEEF sincos
   test_fcn("double AVX2", [&](){ 
      
      int p = 4;
      int nm = N / p;

      __m256d* maskm = (__m256d*) &mask[0];
      __m256d* phasem = (__m256d*) &phase[0];


      #pragma omp parallel for
      for (int i = 0; i<nm; i++)
      {
         m256d2 fieldm;

         fieldm = Sleef_sincosd4_u10avx2(phasem[i]);
         fieldm.x = _mm256_mul_pd(fieldm.x, maskm[i]);
         fieldm.y = _mm256_mul_pd(fieldm.y, maskm[i]);

         double* f = (double*)&field[i*p];
         for (int j = 0; j < p; j++)
         {
            *(f++) = fieldm.y.m256d_f64[j];
            *(f++) = fieldm.x.m256d_f64[j];
         }
      }

   });

   // single AVX2 using SLEEF sincos
   test_fcn("single AVX2", [&](){ 
      
      int p = 8;
      int nm = N / p;

      __m256* maskm = (__m256*) &mask_f[0];
      __m256* phasem = (__m256*) &phase_f[0];

      #pragma omp parallel for
      for (int i = 0; i < nm; i++)
      {
         m256s2 fieldm;
         fieldm = Sleef_sincosf8_u10avx2(phasem[i]);

         fieldm.x = _mm256_mul_ps(fieldm.x, maskm[i]);
         fieldm.y = _mm256_mul_ps(fieldm.y, maskm[i]);

         double* f = (double*)&field_f[i * p];
         for (int j = 0; j < p; j++)
         {
            *(f++) = fieldm.y.m256_f32[j];
            *(f++) = fieldm.x.m256_f32[j];
         }
      }


   });
   
   // Check AVX results against baseline
   for (int i = 0; i < N; i++)
      if (abs(field[i] - field_ref[i]) > 1e-6)
         throw std::runtime_error("Incorrect results");

   return 0;
}