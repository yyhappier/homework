#include<stdio.h>
#include<math.h>

#define n 4

void Doolittle(double a[n][n],double *b)
{
    int i=0,j=0,k=0,r=0;
    double u[n][n]={0},l[n][n]={0},y[n],x[n],norm2,Ax[n],aa[n][n],bb[n];
    //algorithm
    //initialize
    for(i=0;i<n;i++)
    	l[i][i]=1;
    	
  	for(k=0;k<n;k++)
  	{
  		//calculate U
  		for(j=k;j<n;j++)
  		{
  			u[k][j]=a[k][j];
			for(r=0;r<=k-1;r++)
  				u[k][j]-=l[k][r]*u[r][j];
  		}
  		//calculate L
  		for(i=k+1;i<n;i++)
  		{
  			l[i][k]=a[i][k];
  			for(r=0;r<=k-1;r++)
  				l[i][k]-=l[i][r]*u[r][k];
  			l[i][k]/=u[k][k];
		}
	  }
	//solve LY=b
	for(i=0;i<n;i++)
	{
		y[i]=b[i];
		for(j=0;j<=i-1;j++)
			y[i]-=l[i][j]*y[j];
	}
	//solve UX=Y
	for(i=n-1;i>=0;i--)
	{
		x[i]=y[i];
		for(j=i+1;j<=n;j++)
			x[i]-=u[i][j]*x[j];
		x[i]=x[i]/u[i][i];
	}
    //print L
    printf("matrix L:\n");
    for(i=0;i<n;i++)
    {
    	for(j=0;j<n;j++)
    	{
			printf("%.6f   ",l[i][j]);
    	}
    	printf("\n");
	}
	//print U
	printf("\nmatrix U:\n");
	for(i=0;i<n;i++)
    {
    	for(j=0;j<n;j++)
    	{
			printf("%.6f   ",u[i][j]);
    	}
    	printf("\n");
	}
    //print result
    printf("\nresult:\n");
    for(i=0;i<n;i++)
        printf("x[%d]=%.6f\n",i+1,x[i]);
        //calculate error
    for(i=0;i<n;i++)
    {
    	for(j=0;j<n;j++)
    		Ax[i]+=aa[i][j]*x[j];
//    	printf("Ax[i]:%.6f ",Ax[i]);
	}
	for(i=0;i<n;i++)
		norm2=pow(Ax[i]-bb[i],2);
	norm2=sqrt(norm2);
	printf("\n2-norm:%.6f",norm2);
}

void main()
{
	//input
	double a[4][4]={
			{7.2,2.3,-4.4,0.5},
			{1.3,6.3,-3.5,2.8},
			{5.6,0.9,8.1,-1.3},
			{1.5,0.4,3.7,5.9}
			};
	double b[4]={15.1,1.8,16.6,36.9};
    Doolittle(a,b);
}