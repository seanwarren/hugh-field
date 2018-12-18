//#define __SSE2__ 1
//#define __AVX__ 1

#define M_PI 3.14159265358979323846264338327950288

#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <random>
#include <boost/align/aligned_allocator.hpp>
#include <sleef.h>
#include <omp.h>


using namespace std;
using namespace std::literals::complex_literals;

template<typename real_iter, typename complex_iter>
void field_0(const real_iter mask, 
             const real_iter phase, 
             complex_iter field,
             int n)
{
   for(int i=0; i<n; i++)
      field[i] = mask[i] * exp(1i * phase[i]);
};

template<typename real_iter, typename complex_iter>
void field_1(real_iter mask, 
             real_iter phase, 
             complex_iter field,
             int n)
{
   #pragma omp parallel for
   for(int i=0; i<n; i++)
      field[i] = mask[i] * exp(1i * phase[i]);
};

typedef Sleef___m256d_2 m256d2;
typedef Sleef___m256_2 m256s2;

void field_2(const vector<double>::iterator mask, 
             const vector<double>::iterator phase, 
             vector<complex<double>>::iterator field,
             int n)
{

   int p = 4;
   int nm = n / p;

   __m256d* maskm = (__m256d*) &mask[0];
   __m256d* phasem = (__m256d*) &phase[0];
   

   #pragma omp parallel for
   for(int i=0; i<nm; i++)
   {
      m256d2 fieldm;
      
      fieldm = Sleef_sincosd4_u10avx2(phasem[i]);
      fieldm.x = _mm256_mul_pd(fieldm.x, maskm[i]);
      fieldm.y = _mm256_mul_pd(fieldm.y, maskm[i]);

      double* f = (double*) &field[i*p];
      for (int j = 0; j < p; j++)
      {
         *(f++) = fieldm.y.m256d_f64[j];
         *(f++) = fieldm.x.m256d_f64[j];
      }


   }
};


void field_3(const vector<float>::iterator mask,
   const vector<float>::iterator phase,
   vector<complex<float>>::iterator field,
   int n)
{

   int p = 8;
   int nm = n / p;

   __m256* maskm = (__m256*) &mask[0];
   __m256* phasem = (__m256*) &phase[0];


   #pragma omp parallel for
   for (int i = 0; i<nm; i++)
   {
      m256s2 fieldm;
      fieldm = Sleef_sincosf8_u10avx2(phasem[i]);
      
      fieldm.x = _mm256_mul_ps(fieldm.x, maskm[i]);
      fieldm.y = _mm256_mul_ps(fieldm.y, maskm[i]);

      double* f = (double*)&field[i * p];
      for (int j = 0; j < p; j++)
      {
         *(f++) = fieldm.y.m256_f32[j];
         *(f++) = fieldm.x.m256_f32[j];
      }


   }
};


int main()
{
   int N = 1024 * 1024;
   int r = 100;

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
      mask[i] = (double) mask_distribution(gen) + 1;

      phase_f[i] = phase[i];
      mask_f[i] = mask[i];
   }

   // Reference zero order calculation
   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<r; i++)
         field_0(mask.begin(), phase.begin(), field_ref.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_0: " << chrono::duration <double> (diff).count()  << " s" << endl;
   }  

   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<r; i++)
         field_1(mask.begin(), phase.begin(), field.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_1: " << chrono::duration <double> (diff).count()  << " s" << endl;
   }  

   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<r; i++)
         field_2(mask.begin(), phase.begin(), field.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_2: " << chrono::duration <double> (diff).count()  << " s" << endl;
   }  

   {
      auto start = chrono::steady_clock::now();
      for (int i = 0; i<r; i++)
         field_3(mask_f.begin(), phase_f.begin(), field_f.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_2: " << chrono::duration <double>(diff).count() << " s" << endl;
   }


   for (int i = 0; i < N; i++)
      if (abs(field[i] - field_ref[i]) > 1e-6)
         throw std::runtime_error("Incorrect results");

   return 0;
}