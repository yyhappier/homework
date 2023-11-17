#include <iostream>
#include <complex>
#include <vector>
# include<math.h>
#include<iomanip>
using namespace std ;
#define PI 3.1415926535

complex<double>* FFT(complex<double> f[],int n)
{
	int i=0,k=0;
	complex<double> f0[n/2],f1[n/2];
	complex<double> *g0,*g1,*g;
	g0=new complex<double>[n/2];
	g1=new complex<double>[n/2];
	g=new complex<double>[n];
	complex<double> w(1,0);
	complex<double> wn=exp(-2.0*PI/n*complex<double>(0.0,1.0));
	if(n==1)
		return f;
	else
	{
		for(i=0;i<n/2;i++)
		{
			f0[i]=f[2*i];
			f1[i]=f[2*i+1];
		}
		g0=FFT(f0,n/2);
		g1=FFT(f1,n/2);
		for(k=0;k<n/2;k++)
		{
			g[k]=(g0[k]+w*g1[k])/2.0;
			g[k+n/2]=(g0[k]-w*g1[k])/2.0;
			w=w*wn;
		}
	}
	return g;
}

complex<double>* IFFT(complex<double> f[],int n)
{
	int i=0,k=0;
	complex<double> f0[n/2],f1[n/2];
	complex<double> *g0,*g1,*g;
	g0=new complex<double>[n/2];
	g1=new complex<double>[n/2];
	g=new complex<double>[n];
	complex<double> w(1,0);
	complex<double> wn=exp(2.0*PI/n*complex<double>(0.0,1.0));
	if(n==1)
		return f;
	else
	{
		for(i=0;i<n/2;i++)
		{
			f0[i]=f[2*i];
			f1[i]=f[2*i+1];
		}
	
		g0=IFFT(f0,n/2);
		g1=IFFT(f1,n/2);
		for(k=0;k<n/2;k++)
		{
			g[k]=(g0[k]+w*g1[k]);
			g[k+n/2]=(g0[k]-w*g1[k]);
			w=w*wn;
		}
	}
	return g;
}


complex<double> F(double t)
{
	complex<double> f=complex<double>(0.7*sin(2*PI*2*t)+sin(2*PI*5*t),0);
	return f;
}

double GetLength(complex<double> gi)
{
	double length;
	length=sqrt(gi.real()*gi.real()+gi.imag()*gi.imag());
	return length;
}

int main()
{
	//	int n=pow(2,4);
	int n=pow(2,7);
	cout<<setiosflags(ios::fixed)<<setprecision(15);
	int i=0;
	double length[n];
	complex<double> f[n];
	for(i=0;i<n;i++)
		f[i]=F(double(i)/n);
	complex<double> *g;
	complex<double>* g_inverse;
	g=new complex<double>[n];
	g_inverse=new complex<double>[n];
	g=FFT(f,n);
	g_inverse=IFFT(g,n);
	for(i=0;i<n;i++)
	{
		length[i]=GetLength(g[i]);
	}
	cout<<"i\treal\t\t\timag\t\t\tlength"<<endl;
//	cout<<"length"<<endl;
	for(i=0;i<n;i++)
		cout<<i<<"\t"<<g[i].real()<<"\t"<<g[i].imag()<<"\t"<<length[i]<<endl;
//		cout<<length[i]<<endl;
	cout<<"f"<<endl;
	for(i=0;i<n;i++) 
		cout<<f[i].real()<<endl;
	cout<<"g_inverse"<<endl;
	for(i=0;i<n;i++) 
		cout<<g_inverse[i].real()<<endl;
	return 0;
}