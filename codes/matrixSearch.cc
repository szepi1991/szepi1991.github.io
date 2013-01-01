/*
 * Compile as: g++ matrixSearch.cc -lpthread -msse4.2 -O3 
 */

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cctype>
#include <fcntl.h>

// SSE instructions
#include <smmintrin.h>
// Use pthread.
#include <pthread.h>
#define NUM_THREADS 40

#define BMAX 8192
#define MAXCOLS 35000 // > 8192 * 4
#define SMAX 16
#define NUM_LINES_TO_READ ((BMAX / NUM_THREADS) + 1)
#define THRESHOLD 128
#define LUTSIZE 101

char bMatrixLow[BMAX][BMAX];
char bMatrixHigh[BMAX][BMAX];

char sMatrix[SMAX][SMAX];
char inputFile[BMAX][MAXCOLS];
double wtime(void)
{
  double sec;
  struct timeval tv;

  gettimeofday(&tv,NULL);
  sec = tv.tv_sec + tv.tv_usec/1000000.0;
  return sec;
}

void *ParallelRead(void *threadId)
{
  const int8_t lutLow[LUTSIZE] = {0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 24, 25, 26, 27, 28, 28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 44, 45, 46, 47, 48, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 64, 65, 66, 67, 68, 68, 69, 70, 71, 72, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80}; // floor([0-255] * 0.8)

  const int8_t lutHigh[LUTSIZE] = {0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120}; // ceiling([0-255]*1.2)

  int start = ((long)threadId) * NUM_LINES_TO_READ;
  int end = start + NUM_LINES_TO_READ + 1;
  if (end > BMAX)
    end = BMAX;
  for (int curr = start; curr < end; curr++)
  {
    char* ptr = inputFile[curr];
    int idx = 0;
    int num = 0;
    while (*ptr)
    {
      if (isdigit(*ptr))
        num = (num * 10) + *ptr - '0';
      else
      {
        bMatrixLow[curr][idx] = lutLow[num]; 
        bMatrixHigh[curr][idx++] = lutHigh[num];
        num = 0;
      }
      ptr++;
    }
  }

    pthread_exit(NULL);
}

void *ParallelSearch(void *threadId)
{
  for (int i = (long)threadId; i <= (BMAX - SMAX); i += NUM_THREADS)
  {
    for (int j = 0; j <= (BMAX - SMAX); j++)
    {
      // Check for match.
      int numMatches = 0;
      int numMisMatches = 0;
      for (int ii = 0; ii < SMAX; ii++)
      {
        char *sLoc = sMatrix[ii];
        char *lLoc = bMatrixLow[i + ii] + j; 
        char *hLoc = bMatrixHigh[i + ii] + j; 
        
        __m128i high, low, num;
        low = _mm_loadu_si128((__m128i*)lLoc);
        high = _mm_loadu_si128((__m128i*)hLoc);
        num = _mm_loadu_si128((__m128i*)sLoc);

        __m128i result = _mm_cmpgt_epi8(num, low);
        result = _mm_and_si128(result, _mm_cmpgt_epi8(high, num));
        int mask = _mm_movemask_epi8(result);
        int count = _mm_popcnt_u32(mask);
        numMatches += count;
        numMisMatches += (SMAX - count);
        
        if (numMisMatches > THRESHOLD)
          break;
      }

      if (numMatches >= THRESHOLD)
          printf("%d.%d\n", i, j);
    }
  }
}

void ReadSmallerFile(FILE* fp)
{
    printf("val:%d\n", sMatrix[15][15]);
}

int main(int argc, char* argv[])
{
    double time = wtime();
    
    FILE* fp = fopen(argv[1], "r");
    int lineNum = 0;
    while (fgets (inputFile[lineNum++], MAXCOLS, fp));
    fclose(fp);
    pthread_t threads[NUM_THREADS];
    long t;
    int rc;
    // Read
    for (t = 0; t < NUM_THREADS; t++){
      pthread_create(&threads[t], NULL, ParallelRead, (void *)t);
    }

    for(t=0; t< NUM_THREADS; t++)
      pthread_join(threads[t], NULL);
    
    // Read smaller file
    fp = fopen(argv[2], "r");
    char sMatFile[SMAX][SMAX*4];
    lineNum = 0;
    while (fgets (sMatFile[lineNum++], 128, fp));
    for (int curr = 0; curr < SMAX; curr++)
    {
      char* ptr = sMatFile[curr];
      int idx = 0;
      int num = 0;
      while (*ptr)
      {
        if (isdigit(*ptr))
          num = (num * 10) + *ptr - '0';
        else
        {
          sMatrix[curr][idx++] = num;
          num = 0;
        }
        ptr++;
      }
    }
    fclose(fp);

    // Search
    for (t = 0; t < NUM_THREADS; t++){
      pthread_create(&threads[t], NULL, ParallelSearch, (void *)t);
    }

    for(t=0; t< NUM_THREADS; t++)
      pthread_join(threads[t], NULL);
    
    printf("Total Program Runtime: %.5lf\n", wtime() - time);
}
