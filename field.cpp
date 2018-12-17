#include <complex>
#include <vector>
#include <chrono>
#include <iostream>
#include <x86intrin.h>
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

void field_2(const vector<__m256d>::iterator mask, 
             const vector<__m256d>::iterator phase, 
             vector<m256d2>::iterator field,
             int n)
{
   #pragma omp parallel for
   for(int i=0; i<n; i++)
   {
      field[i] = Sleef_sincosd4_u10avx2(mask[i]);
      //field[i] = mask[i] * exp(1i * phase[i]);
   }
};


int main()
{
   int N = 1024 * 1024;
   int r = 10;

   std::vector<double> mask(N);
   std::vector<double> phase(N);
   std::vector<complex<double>> field(N);
   
   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<r; i++)
         field_0(mask.begin(), phase.begin(), field.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_0: " << chrono::duration <double, milli> (diff).count()  << " ms" << endl;
   }  

   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<r; i++)
         field_1(mask.begin(), phase.begin(), field.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_1: " << chrono::duration <double, milli> (diff).count()  << " ms" << endl;
   }  

   {
      auto start = chrono::steady_clock::now();
      for(int i=0; i<r; i++)
         field_1(mask.begin(), phase.begin(), field.begin(), N);
      auto diff = chrono::steady_clock::now() - start;
      cout << "field_1: " << chrono::duration <double, milli> (diff).count()  << " ms" << endl;
   }  


   return 0;
}