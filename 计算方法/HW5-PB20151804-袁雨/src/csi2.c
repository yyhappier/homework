#include<stdio.h>
#include<math.h>

#define n 21
double m[n];
//double fx(double x)
//{
//	double fx=x/(x*x+x+1);
//	return fx;
//}
//
//double ffx(double x)
//{
//	double ffx=(1-x*x)/((x*x+x+1)*(x*x+x+1));
//	return ffx;
//}

double fx(double x)
{
	double fx=(x+sin(2*x))/(1+exp(-x));
	return fx;
}

double ffx(double x)
{
	double ffx=exp(x)*((exp(x)+x+sin(2*x))+2*(exp(x)+1)*cos(2*x)+1)/((exp(x)+1)*(exp(x)+1));
	return ffx;
}

double Thomas(double a[n][n],double f[n])
{
	double u[n]={0},v[n]={0},c[n]={0},b[n]={0},y[n]={0};
	int i=0,k=0;
	for(k=0;k<n;k++)
	{
		c[k+1]=a[k+1][k];
		b[k]=a[k][k+1];
	}
	//decompose
	for(k=0;k<n;k++)
	{
		u[k]=a[k][k]-c[k]*v[k-1];
		v[k]=b[k]/u[k];
	}
	//solve Ly=f
	for(i=0;i<n;i++)
	{
		y[i]=(f[i]-c[i]*y[i-1])/u[i];
	}
	//Solve Ux=y;
	for(i=n-1;i>=0;i--)
	{
		if(i<n-1)
			m[i]=y[i]-v[i]*m[i+1];
		else
			m[i]=y[i];
	}
}

void main()
{
	double h[n]={0},lambda[n]={0},mu[n]={0},M[n]={0},d[n];
//	double x[n]={-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0};
//	double x[n]={-2.0,-1.88,-1.63,-1.52,-1.24,-1.0,-0.92,-0.73,-0.41,-0.33,0,0.25,0.33,0.55,0.88,1.12,1.23,1.45,1.69,1.82,2.0};
//	double x[n]={-2.0,-1.83,-1.62,-1.5,-1.2,-1,-0.88,-0.75,-0.66,-0.53,-0.42,-0.2,-0.11,0.05,0.25,0.4,0.6,0.88,1.22,1.62,2.0};

//	double x[n]={-2.0,-1.7,-1.4,-1.1,-0.8,-0.5,-0.2,0.1,0.4,0.7,1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1,3.4,3.7,4.0};
//	double x[n]={-2.0,-1.85,-1.66,-1.2,-0.8,-0.66,-0.24,0.15,0.44,0.66,1.0,1.33,1.56,1.95,2.12,2.55,2.81,3.01,3.49,3.72,4.0};
	double x[n]={-2.0,-1.85,-1.56,-1.2,-0.98,-0.76,-0.54,-0.35,-0.14,0.56,0.88,1.13,1.56,1.88,2.12,2.33,2.45,2.66,3.42,3.8,4.0};
	
	double y[n]={0};
	double a[n][n],f[n],m0,mn;
	char Sx[n];
	m0=ffx(x[0]);
	mn=ffx(x[n]);
	double temp,temp1,temp2;
	int i=0;
	for(i=0;i<n;i++)
		h[i]=x[i+1]-x[i];
	d[0]=6.0/h[0]*((fx(x[1])-fx(x[0]))/h[0]-m0);
	d[n]=6.0/h[0]*(mn-(fx(x[n])-fx(x[n-1]))/h[n-1]);
	for(i=1;i<n;i++)
	{
		lambda[i]=h[i]/(h[i]+h[i-1]);
		mu[i]=1-lambda[i];
		d[i]=6.0/(h[i]+h[i-1])*((fx(x[i+1])-fx(x[i]))/h[i]-(fx(x[i])-fx(x[i-1]))/h[i-1]);
	}
	for(i=0;i<n;i++)
	{
		a[i][i]=2;
		a[0][1]=1;
		a[n][n-1]=1;
		a[i+1][i+2]=lambda[i+1];
		a[i+1][i]=mu[i+1];
		f[i]=d[i];
	}
	Thomas(a,f);
	for(i=0;i<n;i++)
	{
		temp=fx(x[i]);
		y[i]=temp;
		printf("(%.6f,%.6f)\n",x[i],y[i]); 
	}
	for(i=0;i<=n;i++)
	{
		M[i]=m[i];
	}
	for(i=0;i<n-1;i++)
	{
		printf("\nSx[%d]:",i);
		printf("\n((%.6f-x)^(3)*(%.6f)+(x-(%.6f))^(3)*(%.6f))/(6*(%.6f)) + ((%.6f-x)*(%.6f)+(x-(%.6f))*(%.6f))/(%.6f)-(%.6f)/6*((%.6f-x)*(%.6f)+(x-(%.6f))*(%.6f)),%.2f<x<%.2f",x[i+1],M[i],x[i],M[i+1],h[i],x[i+1],y[i],x[i],y[i+1],h[i],h[i],x[i+1],M[i],x[i],M[i+1],x[i],x[i+1]);
//		temp2=(x[i+1]*y[i]-x[i]*y[i+1])/0.2;
//		temp1=(y[i+1]-y[i])/0.2;
//		printf("\n%.6fx+(%.6f)",temp1,temp2);
	}
}