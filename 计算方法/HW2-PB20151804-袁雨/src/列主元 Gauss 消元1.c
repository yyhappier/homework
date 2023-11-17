#include<stdio.h>
#include<math.h>

#define n 5

void GaussPP(double a[n][n],double *b)
{
    int i=0,j=0,k=0;
    double t,norm2,Ax[n],aa[n][n],bb[n];
    //copy
    for(i=0;i<n;i++)
    {
    	bb[i]=b[i];
    	for(j=0;j<n;j++)
    		aa[i][j]=a[i][j];
	}
    //algorithm
    for(i=0;i<n;i++)
    {
    	//select max
        k=i;
        for(j=i+1;j<n;j++)
        {
            if (abs(a[k][i])<abs(a[j][i]))
                k=j;
        }
        //exchage
        for(j=i;j<n;j++)
        {
            t=a[i][j];
            a[i][j]=a[k][j];
            a[k][j]=t;
        }
        t=b[i];
        b[i]=b[k];
        b[k]=t;
        //eliminate
        for(j=i+1;j<n;j++)
        {
            a[j][i]=a[j][i]/a[i][i];
            for(k=i+1;k<n;k++)
            {
                a[j][k]=a[j][k]-a[j][i]*a[i][k];
            }
            b[j]=b[j]-a[j][i]*b[i];
        }
    }
    //print U
    printf("matrix U:\n");
    for(i=0;i<n;i++)
    {
    	for(j=0;j<n;j++)
    	{
			if(i>j)
				printf("0.000000   ");
    		else
				printf("%.6f   ",a[i][j]);
    	}
    	printf("\n");
	}
    //result
    for(i=n-1;i>=0;i--)
    {
        for(j=i+1;j<n;j++)
            b[i]=b[i]-a[i][j]*b[j];
        b[i]=b[i]/a[i][i];
    }
    printf("\nresult:\n");
    for(i=0;i<n;i++)
        printf("x[%d]=%.6f\n",i+1,b[i]);
    //calculate error
    for(i=0;i<n;i++)
    {
    	for(j=0;j<n;j++)
    		Ax[i]+=aa[i][j]*b[j];
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
	double a[n][n]={
			{1.0/9.0,1.0/8.0,1.0/7.0,1.0/6.0,1.0/5.0},
			{1.0/8.0,1.0/7.0,1.0/6.0,1.0/5.0,1.0/4.0},
			{1.0/7.0,1.0/6.0,1.0/5.0,1.0/4.0,1.0/3.0},
			{1.0/6.0,1.0/5.0,1.0/4.0,1.0/3.0,1.0/2.0},
			{1.0/5.0,1.0/4.0,1.0/3.0,1.0/2.0,1.0/1.0}
			};
	double b[n]={1,1,1,1,1};
//    printf("please input n:");
//    scanf("%d",&n);
//    double a[n+1][n+1],b[n+1];
//    printf("please input a[%d][%d]:\n",n,n);
//    for(i=1;i<=n;i++)
//    {
//        for(j=1;j<=n;j++)
//            scanf("%lf",&a[i][j]);
//    }
//    printf("please input b[%d]:\n",n);
//    for(i=1;i<=n;i++)
//    {
//        scanf("%lf",&b[i]);
//    }
    GaussPP(a,b);
}
