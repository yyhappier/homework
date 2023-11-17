#include<stdio.h>
#include<math.h>
float f(float x)
{
	float fx;
	fx=pow((x-1),3)-pow(x,2)+x;
	return fx;
}
float ff(float x)
{
	float ffx;
	ffx=3*pow((x-1),2)-2*x+1;
	return ffx;
}
float g(float x)
{
	float gx;
	gx=x-f(x)/ff(x);
	return gx;
}

int main()
{
	int i=0;
	float x0,x1,e;
	e=1.0e-5;
	printf("input x0=");
	scanf("%f",&x0);
	while(i<10000)
	{	x1=g(x0);
		i++;
		if(fabs(x1-x0)<e)
		{
			printf("approximate root:%.12f\n",x1);
			return 0;
		}
		printf("iteration:%d:f(%.12f)=%.12f\n",i,x1,f(x1));
		x0=x1;
	}
	printf("there is no root near x0!");
	return 1;
}