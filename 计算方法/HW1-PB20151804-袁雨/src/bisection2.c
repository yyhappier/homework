#include<stdio.h>
#include<math.h>
float f(float x)
{
	float fx;
	fx=pow(sin(x),3)+pow(cos(x),3);
	return fx;
}
void main()
{
	int i=0;
	float a,b,x,e;
	e=1.0e-5;
	printf("input a=");
	scanf("%f",&a);
	printf("input b=");
	scanf("%f",&b);
	if(f(a)*f(b)>0)
		printf("please choose other methods!");
	else
		while(1)
		{
			x=(a+b)/2;
			i++;
			if(fabs(f(x))<e)
			{	
				printf("iteration:%d:f(%.12f)=%.12f\n",i,x,f(x));
				printf("approximate root:%.12f\n",x);
				break;
			}
			else if(f(a)*f(x)<0)
				b=x;
			else if(f(b)*f(x)<0)
				a=x;
			printf("iteration:%d:f(%.12f)=%.12f\n",i,x,f(x));
		}
}