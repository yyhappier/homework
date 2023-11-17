#include<stdio.h>
#include<math.h>

#define n 5
#define epsilon 1e-5

double x[1000][n]={1.0,1.0,1.0,1.0,1.0},y[1000][n]={1.0,1.0,1.0,1.0,1.0};
//x[0]={1.0,1.0,1.0,1.0,1.0};
//y[0]={1.0,1.0,1.0,1.0,1.0};
int m=0;

double max(double x[n])
{
	double max=0.0;
	for(int i=0;i<n;i++)
	{
		if(fabs(max)<fabs(x[i]))
			max=(x[i]);
	}
	return max;
}

void Doolittle(double a[n][n],double *b)
{
	m++;
    int i=0,j=0,k=0,r=0;
    double u[n][n]={0},l[n][n]={0},yy[n],Ax[n];
    //algorithm
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
		yy[i]=b[i];
		for(j=0;j<=i-1;j++)
			yy[i]-=l[i][j]*yy[j];
	}
	//solve UX=Y
	for(i=n-1;i>=0;i--)
	{	
		x[m][i]=yy[i];
		for(j=i+1;j<=n;j++)
			x[m][i]-=u[i][j]*x[m][j];
		x[m][i]=x[m][i]/u[i][i];
	}
	for(i=0;i<n;i++)
		{
			y[m][i]=x[m][i]/fabs(max((x[m])));
		}
}


void Inverse(double a[n][n],double b[n])
{
	//input
	double lambda1,lambda2,v1[n],v2[n];
	Doolittle(a,y[m]);
	Doolittle(a,y[m]);
	while((fabs(max(x[m])-max(x[m-1]))>epsilon)
			&&(fabs((max(x[m])-max(x[m-2])))>epsilon)
			&&(m<1000))
	{
		Doolittle(a,y[m]);
	}
	if(fabs(max(x[m])-max(x[m-1]))<=epsilon)
	{
		if(max(x[m])*max(x[m-1])>0)
		{
			lambda1=1/fabs(max(x[m]));
			printf("lambda1:%.6f\n",lambda1);
			printf("v1:(");
			for(int i=0;i<n;i++)
			{
				v1[i]=y[m][i];
				if(i!=n-1)
    				printf("%.6f,",v1[i]);
    			else
    				printf("%.6f)",v1[i]);
			}
		}
		else if(max(x[m])*max(x[m-1])<0)
		{
			lambda1=-1/fabs(max(x[m]));
			printf("lambda1:%.6f\n",lambda1);
			printf("v1:(");
			for(int i=0;i<n;i++)
			{
				v1[i]=y[m][i];
				if(i!=n-1)
    				printf("%.6f,",v1[i]);
    			else
    				printf("%.6f)",v1[i]);
			}

		}
	}
	else if(fabs((max(x[m])-max(x[m-2])))<=epsilon)
	{
		m++;
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
				x[m][i]+=a[i][j]*x[m-1][j];
		}
		lambda1=1/sqrt(fabs(max(x[m])/max(y[m-2])));
		printf("lambda1:%.6f\n",lambda1);
		lambda2=-lambda1;
		printf("lambda2:%.6f\n",lambda2);
		for(int i=0;i<n;i++)
		{
			v1[i]=x[m][i]+lambda1*x[m-1][i];
			v2[i]=x[m][i]-lambda1*x[m-1][i];
		}
		printf("v1:(");
		for(int i=0;i<n;i++)
		{
			if(i!=n-1)
    			printf("%.6f,",v1[i]);
    		else
    			printf("%.6f)",v1[i]);
		}
		printf("\nv2:(");
		for(int i=0;i<n;i++)
		{
			if(i!=n-1)
    			printf("%.6f,",v2[i]);
    		else
    			printf("%.6f)",v2[i]);
		}
	}
	else
		printf("please choose another method!");
	printf("\n");
}

void main()
{
	//input
	double a[n][n]={
			{1.0/9.0,1.0/8.0,1.0/7.0,1.0/6.0,1.0/5.0},
			{1.0/8.0,1.0/7.0,1.0/6.0,1.0/5.0,1.0/4.0},
			{1.0/7.0,1.0/6.0,1.0/5.0,1.0/4.0,1.0/3.0},
			{1.0/6.0,1.0/5.0,1.0/4.0,1.0/3.0,1.0/2.0},
			{1.0/5.0,1.0/4.0,1.0/3.0,1.0/2.0,1.0/1.0}
			};
	double b[n]={1,1,1,1,1};
    Inverse(a,b);
    for(int k=0;k<=m;k++)
    {
    	//the first condition
    	printf("lambda1(%d):%.6f",k,1/fabs(max(x[k])));
    	printf("\tX(%d):(",k);
    	for(int i=0;i<n;i++)
    	{
    		if(i!=n-1)
    			printf("%-13.6f,",x[k][i]);
    		else
    			printf("%-13.6f)",x[k][i]);
		}
		if(k!=m)
		{
			printf("\tY(%d):(",k);
    		for(int i=0;i<n;i++)
    		{
    			if(i!=n-1)
    				printf("%-10.6f,",y[k][i]);
    			else
    				printf("%-10.6f)",y[k][i]);
			}
			printf("\n");
		}
	}
}