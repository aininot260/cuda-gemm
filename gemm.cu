#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define eps 1e-2
#define TILE_WIDTH 32  //89*89*4*2=63368B<64KB

int m,n,k;
double duration;

__global__ void MatrixMulKernle(int m, int n, int k, float *A,float  *B, float *C)
{
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
 
	int bx = blockIdx.x;		int by = blockIdx.y;
	int tx = threadIdx.x;		int ty = threadIdx.y;
 
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
 
	float Cvalue = 0;
 
	for (int t=0; t<(n-1)/TILE_WIDTH+1; ++t)
	{
		if (Row < m && t * TILE_WIDTH + tx < n)
		    ds_A[tx][ty] = A[Row*n+t*TILE_WIDTH+tx];
		else
			ds_A[tx][ty] = 0.0;
 
		if (t * TILE_WIDTH + ty < n && Col < k)
            ds_B[tx][ty] = B[(t*TILE_WIDTH + ty)*k+Col];
		else
			ds_B[tx][ty] = 0.0;	
 
		__syncthreads();
		
		for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += ds_A[i][ty] * ds_B[tx][i];
 
		__syncthreads();
 
		if(Row < m && Col < k)
			C[Row*k+Col]=Cvalue;		
	}
}

float ds_A[TILE_WIDTH][TILE_WIDTH]; 
float ds_B[TILE_WIDTH][TILE_WIDTH];
void MatrixMulKernleSerialize(int m, int n, int k, float *A,float  *B, float *C)
{
	for(int bx=0;bx<(k-1)/TILE_WIDTH+1;bx++)
	{
		for(int by=0;by<(m-1)/TILE_WIDTH+1;by++)
		{
			for (int t=0; t<(n-1)/TILE_WIDTH+1; ++t)
			{
				for(int tx=0;tx<TILE_WIDTH;tx++)
				for(int ty=0;ty<TILE_WIDTH;ty++)
				{
					int Row = by * TILE_WIDTH + ty;
					int Col = bx * TILE_WIDTH + tx;

					if (Row < m && t * TILE_WIDTH + tx < n)
						ds_A[tx][ty] = A[Row*n+t*TILE_WIDTH+tx];
					else
						ds_A[tx][ty] = 0.0;
			
					if (t * TILE_WIDTH + ty < n && Col < k)
						ds_B[tx][ty] = B[(t*TILE_WIDTH + ty)*k+Col];
					else
						ds_B[tx][ty] = 0.0;	
				}
				for(int tx=0;tx<TILE_WIDTH;tx++)
				for(int ty=0;ty<TILE_WIDTH;ty++)
				{
					int Row = by * TILE_WIDTH + ty;
					int Col = bx * TILE_WIDTH + tx;

					float Cvalue = 0;
	
					for (int i = 0; i < TILE_WIDTH; ++i)
						Cvalue += ds_A[i][ty] * ds_B[tx][i];
			
					if(Row < m && Col < k)
						C[Row*k+Col]+=Cvalue;		
				}
			}
		}
	}
}

void MatrixMulSample(int m, int n, int k, float* A, float* B, float* C)
{
    for (int Row = 0; Row < m; ++ Row)
		for (int Col = 0; Col < k; ++ Col )
		{
			float sum = 0;
			for (int i = 0; i < n; ++i)
			{
				float a = A[Row * n + i];
				float b = B[Col + i * k];
				sum += a * b;
			}
			C[Row* k + Col] = sum;
		}
}
 
int main()
{
	clock_t start, stop;
	srand(time(0));
	m=1001,n=100003,k=987;

	float *A=(float*)malloc(m*n*sizeof(float));
	float *B=(float*)malloc(n*k*sizeof(float));
	float *C=(float*)malloc(m*k*sizeof(float));
	float *sample_C=(float*)malloc(m*k*sizeof(float));

	for(int i=0;i<m*n;i++)
		A[i]=(float)(rand()%100)/100;
	for(int i=0;i<n*k;i++)
		B[i]=(float)(rand()%100)/100;

	//分配显存空间
	int size = sizeof(float);
	float *d_a;
	float *d_b;
	float *d_c;
	cudaMalloc((void**)&d_a,m*n*size);
	cudaMalloc((void**)&d_b,n*k*size);
	cudaMalloc((void**)&d_c,m*k*size);
 
	//把数据从Host传到Device
	cudaMemcpy(d_a, A, size*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size*n*k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, C, size*m*k, cudaMemcpyHostToDevice);
 
	//分配网格结构
	dim3 dimGrid((k-1)/TILE_WIDTH+1,(m-1)/TILE_WIDTH+1,1);	//向上取整
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
 
	start=clock();
	//调用内核函数
	MatrixMulKernle<<<dimGrid,dimBlock>>>(m,n,k,d_a,d_b,d_c);
	cudaDeviceSynchronize();
	//将结果传回到主机端
	stop=clock();
	duration=(double)(stop-start)/CLK_TCK;
	printf("GPU version time used    : %.3lfs\n",duration);

	cudaMemcpy(C, d_c, size*m*k, cudaMemcpyDeviceToHost);

	// start=clock();  //C:k*m 
	// MatrixMulKernleSerialize(m,n,k,A,B,C);
	// stop=clock();
	// duration=(double)(stop-start)/CLK_TCK;
	// printf("Tile version time used:   %.3lfs\n",duration);

	start=clock();
	MatrixMulSample(m,n,k,A,B,sample_C);
	stop=clock();
	duration=(double)(stop-start)/CLK_TCK;
	printf("Normal version time used : %.3lfs\n",duration);

	bool flag=1;
	for (int i=0;i<m*k;i++)
		if(fabs(C[i]-sample_C[i])>eps)
		{
			flag=0;
			printf("%f %f\n",C[i],sample_C[i]);
		}
	if(flag)
		puts("PASS");
	else
		puts("FAILED");

	free(A);
	free(B);
	free(C);
	free(sample_C);

    //释放显存空间
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
 
	return 0;
}