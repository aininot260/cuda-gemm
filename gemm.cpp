#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
vector<pair<int,float> > vec1;
vector<pair<int,float> > vec2;

bool cmp(pair<int,float> x,pair<int,float> y)
{
	if(x.first==y.first)
		return x.second<y.second;
	return x.first<y.first;
}

#define eps 1e-2
#define TILE_WIDTH 89  //89*89*4*2=63368B<64KB

int m,n,k;
double duration;

void transformMatrix2D_CPU(float in[][TILE_WIDTH],float out[][TILE_WIDTH],int nx,int ny)
{
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      out[i][j]=in[j][i];
    }
  }
}

void copyMatrix2D_CPU(float in[][TILE_WIDTH],float out[][TILE_WIDTH],int nx,int ny)
{
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
		out[i][j]=in[i][j];
    }
  }
}

float ds_A[TILE_WIDTH][TILE_WIDTH]; 
float ds_B[TILE_WIDTH][TILE_WIDTH];
float ds_tmp[TILE_WIDTH][TILE_WIDTH];
float zero[TILE_WIDTH];  //shared memory里面

void MatrixMulKernleSerialize(int m, int n, int k, float *A,float  *B, float *C)
{
	for(int bx=0;bx<(k-1)/TILE_WIDTH+1;bx++)
	{
		for(int by=0;by<(m-1)/TILE_WIDTH+1;by++)
		{
			for (int t=0; t<(n-1)/TILE_WIDTH+1; ++t)
			{
				for(int tx=0;tx<TILE_WIDTH;tx++)
				if(by * TILE_WIDTH + tx<m)
				{
					memcpy(ds_A[tx],A + (by * TILE_WIDTH + tx)* n+t*TILE_WIDTH,TILE_WIDTH*sizeof(float));
					if(n-t*TILE_WIDTH<TILE_WIDTH)
						memcpy(ds_A[tx]+n-t*TILE_WIDTH,zero,(TILE_WIDTH-(n-t*TILE_WIDTH))*sizeof(float));
				}
				else
					memcpy(ds_A[tx],zero,TILE_WIDTH*sizeof(float));

				// copyMatrix2D_CPU(ds_tmp,ds_A,TILE_WIDTH,TILE_WIDTH);

				for(int tx=0;tx<TILE_WIDTH;tx++)
				if(t * TILE_WIDTH + tx<n)
				{
					memcpy(ds_B[tx],B + bx*TILE_WIDTH+(t * TILE_WIDTH + tx)*k,TILE_WIDTH*sizeof(float));
					if(k-bx*TILE_WIDTH<TILE_WIDTH)
						memcpy(ds_B[tx]+k-bx*TILE_WIDTH,zero,(TILE_WIDTH-(k-bx*TILE_WIDTH))*sizeof(float));
				}
				else
					memcpy(ds_B[tx],zero,TILE_WIDTH*sizeof(float));

				// copyMatrix2D_CPU(ds_tmp,ds_B,TILE_WIDTH,TILE_WIDTH);

				// for(int i=0;i<TILE_WIDTH;i++)
				// for(int j=0;j<TILE_WIDTH;j++)
				// 	vec1.push_back(make_pair(i*TILE_WIDTH+j,ds_B[i][j]));

				// for(int tx=0;tx<TILE_WIDTH;tx++)
				// for(int ty=0;ty<TILE_WIDTH;ty++)
				// {
				// 	int Row = by * TILE_WIDTH + ty;
				// 	int Col = bx * TILE_WIDTH + tx;

				// 	if (Row < m && t * TILE_WIDTH + tx < n)
				// 		ds_tmp[tx][ty] = A[Row*n+t*TILE_WIDTH+tx];
				// 	else
				// 	{
				// 		ds_tmp[tx][ty] = 0.0;
				// 	}
				// }

				// transformMatrix2D_CPU(ds_tmp,ds_A,TILE_WIDTH,TILE_WIDTH);

				// for(int tx=0;tx<TILE_WIDTH;tx++)
				// for(int ty=0;ty<TILE_WIDTH;ty++)
				// {
				// 	int Row = by * TILE_WIDTH + ty;
				// 	int Col = bx * TILE_WIDTH + tx;

				// 	if (t * TILE_WIDTH + ty < n && Col < k)
				// 		ds_tmp[tx][ty] = B[(t*TILE_WIDTH + ty)*k+Col];
				// 	else
				// 		ds_tmp[tx][ty] = 0.0;	
				// }

				// transformMatrix2D_CPU(ds_tmp,ds_B,TILE_WIDTH,TILE_WIDTH);

				// for(int i=0;i<TILE_WIDTH;i++)
				// for(int j=0;j<TILE_WIDTH;j++)
				// 	vec2.push_back(make_pair(i*TILE_WIDTH+j,ds_B[i][j]));

				for(int tx=0;tx<TILE_WIDTH;tx++)
				for(int ty=0;ty<TILE_WIDTH;ty++)
				{
					int Row = by * TILE_WIDTH + ty;
					int Col = bx * TILE_WIDTH + tx;
					
					float Cvalue = 0;

					for (int i = 0; i < TILE_WIDTH; ++i)
						Cvalue += ds_B[i][ty] * ds_A[tx][i];
			
					if((by*TILE_WIDTH+Col-bx*TILE_WIDTH)< m && (bx*TILE_WIDTH+Row-by*TILE_WIDTH) < k)
						C[(by*TILE_WIDTH+Col-bx*TILE_WIDTH)*k+(bx*TILE_WIDTH+Row-by*TILE_WIDTH)]+=Cvalue;		
				}

				// sort(vec1.begin(),vec1.end(),cmp);
				// sort(vec2.begin(),vec2.end(),cmp);

				// bool flag=1;
				// for(int i=0;i<min(vec1.size(),vec2.size());i++)
				// 	if((vec1.size()!=vec2.size())||vec1[i].first!=vec2[i].first||(fabs(vec1[i].second-vec2[i].second)>eps))
				// 		flag=0;
						
				// if(flag)
				// 	cout<<"CHECK ";
				// else
				// 	cout<<"DEL ";

				// if(flag==0)
				// {
				// 	cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
				// 	for(int i=0;i<vec1.size();i++)
				// 		cout<<vec1[i].first<<","<<vec1[i].second<<" ";
				// 	cout<<endl<<"============================== "<<endl;
				// 	for(int i=0;i<vec2.size();i++)
				// 		cout<<vec2[i].first<<","<<vec2[i].second<<" ";
				// 	cout<<endl;
				// 	cout<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
				// 	system("pause");
				// }

				// vec1.clear();
				// vec2.clear();
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
	m=128,n=5760,k=170;

	float *A=(float*)malloc(m*n*sizeof(float));
	float *B=(float*)malloc(n*k*sizeof(float));
	float *C=(float*)malloc(m*k*sizeof(float));
	float *sample_C=(float*)malloc(m*k*sizeof(float));

	for(int i=0;i<m*n;i++)
		A[i]=(float)(rand()%100)/100;
	for(int i=0;i<n*k;i++)
		B[i]=(float)(rand()%100)/100;

	start=clock();  //C:k*m 
	MatrixMulKernleSerialize(m,n,k,A,B,C);
	stop=clock();
	duration=(double)(stop-start)/CLK_TCK;
	printf("Tile version time used :   %.3lfs\n",duration);


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
 
	return 0;
}