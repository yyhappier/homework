#include<stdio.h>
#include<math.h>
#define e 1e-6
#define M 10
#define n 10
#define PI 3.1415926536

double f(double x)
{
//	double f=log(x);
	double f=sqrt(2-(sin(x))*(sin(x)));
	return f;
}


void main()
{
	double h[M+1]={0},RR[M+1][M+1]={0},R[M+1][M+1]={0};
	double fa=0,fb=0,sum=0,differ=1;
	int i=0,j=0,m=0,p=0,k=0;
	double a[n+1],b[n+1],result=0;
//	a[0]=1;
//	b[n-1]=2;
	a[0]=-1.0*PI/6.0;
	b[n-1]=3.0*PI/4.0;
	h[1]=(b[n-1]-a[0])/n;
	double d=(b[n-1]-a[0])/n;
	for(i=0;i<n;i++)
	{
		a[i]=a[0]+i*d;
		b[i]=a[0]+(i+1)*d;
		fa=f(a[i]);
		fb=f(b[i]);
		RR[1][1]=(fa+fb)*h[1]/2.0;
		R[1][1]+=RR[1][1];
		for(k=2;k<=M;k++)
		{
			sum=0;
			h[k]=h[1]/pow(2,k-1);
			for(j=1;j<=pow(2,k-2);j++)
			{
				sum+=f(a[i]+(2*j-1)*h[k]);
			}
			RR[k][1]=(RR[k-1][1]+h[k-1]*sum)/2.0;
			R[k][1]+=RR[k][1];
		}
	}
	for(k=2;(k<=M)&&(differ>=e);k++)
	{
		for(j=2;j<=k;j++)
		{
			R[k][j]=R[k][j-1]+(R[k][j-1]-R[k-1][j-1])/(pow(4,j-1)-1);
		}
		differ=fabs(R[k][k]-R[k-1][k-1]);
	}
	
	for(i=1;i<k;i++)
	{
		for(j=1;j<=i;j++)
		{
			printf("%.10f ",R[i][j]);
		}
		printf("\n");
	}
}