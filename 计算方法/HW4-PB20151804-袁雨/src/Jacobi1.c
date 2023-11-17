#include<stdio.h>
#include<math.h>

#define n 4
#define e 1e-6

int p,q;

double getmax(double a[n][n])
{
	double max=0;
	int i,j;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			if(i!=j)
			{
				if(fabs(a[i][j])>max)
				{
					max=fabs(a[i][j]);
					p=i;
					q=j;
				}	
			}
		}
	}
}

double getsum(double a[n][n])
{
	int i,j;
	double sum;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			if(i!=j)
				sum+=a[i][j]*a[i][j];
		}
	}
	return sum;
}

double getdeter(double a[n][n])
{
	//LU
	int i=0,j=0,k=0,r=0;
    double u[n][n]={0},l[n][n]={0},deter=1.0;
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
//		//print U
//	printf("\nmatrix U:\n");
//	for(i=0;i<n;i++)
//    {
//    	for(j=0;j<n;j++)
//    	{
//			printf("%.10f   ",u[i][j]);
//    	}
//    	printf("\n");
//	}
	  //determinant
	for(i=0;i<n;i++)
	  	deter*=u[i][i];
	return deter;
}

void Jacobi(double a[n][n])
{
	double sum[1000]={0};
	double aa[n][n],x[n][n];
	double s=0.0,t=0.0,t1=0.0,t2=0.0,c=0.0,d=0.0,deter=1,temp1,temp2;
	int i,j,k=0;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			aa[i][j]=a[i][j];
		}
	}
	sum[0]=getsum(a);
	printf(" A(%d):\n",k);
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			printf("%10.6f   ",a[i][j]);
		}
		printf("\n");
	}
	printf("sum(%d):%.7f\n",k,sum[k]);
	while(sum[k]>e)
	{
		k++;
		getmax(a);
		s=(a[q][q]-a[p][p])/(2*a[p][q]);
		if(s==0)
			t=1;
		else
		{
			t1=-s-sqrt(s*s+1);
			t2=-s+sqrt(s*s+1);
			if(fabs(t1)>fabs(t2))
				t=t2;
			else
				t=t1;
		}
		c=1/(sqrt(1+t*t));
		d=t/(sqrt(1+t*t));
		for(i=0;i<n;i++)
		{
			if(i!=p&&i!=q)
			{
				temp1=a[p][i];
				temp2=a[q][i];
				a[i][p]=c*temp1-d*temp2;
				a[p][i]=a[i][p];
				a[i][q]=c*temp2+d*temp1;
				a[q][i]=a[i][q];
			}
		}
		temp1=a[p][p];
		temp2=a[q][q];
		a[p][p]=temp1-t*a[p][q];
		a[q][q]=temp2+t*a[p][q];
		a[p][q]=0;
		a[q][p]=0;
		printf("\n A(%d):\n",k);
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{
				printf("%10.6f   ",a[i][j]);
			}
			printf("\n");
		}
		sum[k]=getsum(a);
		printf("sum(%d):%.7f\n",k,sum[k]);
		
	}
	printf("over!\n");
	printf("\n");
	for(i=0;i<n;i++)
		printf("lambda(%d):%.6f\t\t",i,a[i][i]);
	printf("\n");
	for(k=0;k<n;k++)
	{
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{
				if(i==j)
					x[i][j]=a[k][k]-aa[i][j];
				else
					x[i][j]=-aa[i][j];
			}
		}
		deter=getdeter(x);
		printf("|lambda(%d)*I-A|:%.8f\t",k,deter);
	}
	printf("\n");
//	for(i=0;i<n;i++)
//	{
//		for(j=0;j<n;j++)
//		{
//			printf("%10.6f,",x[i][j]);
//		}
//		printf("\n");
//	}
}

void main()
{
	//input
	double a[n][n]={
			{1,2,3,4},
			{2,5,6,7},
			{3,6,8,9},
			{4,7,9,10}
			};
    Jacobi(a);
}