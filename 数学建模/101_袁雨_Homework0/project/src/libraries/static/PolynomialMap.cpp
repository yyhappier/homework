#include "PolynomialMap.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

using namespace std;

#define EPSILON 1.0e-10

PolynomialMap::PolynomialMap(const PolynomialMap& other) 
{
	m_Polynomial = other.m_Polynomial;
}

PolynomialMap::PolynomialMap(const string& file) 
{
    ReadFromFile(file);
}

PolynomialMap::PolynomialMap(const double* cof, const int* deg, int n) 
{
	for (int i = 0; i < n; i++)
		coff(deg[i]) = cof[i];
}

PolynomialMap::PolynomialMap(const vector<int>& deg, const vector<double>& cof) 
{
	assert(deg.size() == cof.size());
	for (size_t i = 0; i < deg.size(); i++)
		coff(deg[i]) = cof[i];
}

double PolynomialMap::coff(int i) const 
{
	auto target = m_Polynomial.find(i);
	if (target == m_Polynomial.end())
		return 0.;

	return target->second; // you should return a correct value
}

double& PolynomialMap::coff(int i) 
{
	return m_Polynomial[i]; // you should return a correct value
}

void PolynomialMap::compress() 
{
	map<int, double>tmpPoly = m_Polynomial;
	m_Polynomial.clear();
	for (const auto& term : tmpPoly)
	{
		if (fabs(term.second) > EPSILON)
			coff(term.first) = term.second;
	}
}

PolynomialMap PolynomialMap::operator+(const PolynomialMap& right) const 
{
	PolynomialMap poly(right);
	for (const auto& term : m_Polynomial)
		poly.coff(term.first) += term.second;

	poly.compress();
	return poly; // you should return a correct value
}

PolynomialMap PolynomialMap::operator-(const PolynomialMap& right) const 
{
	PolynomialMap poly(right);
	for (const auto& term : m_Polynomial)
		poly.coff(term.first) -= term.second;

	poly.compress();
	return poly; // you should return a correct value
}

PolynomialMap PolynomialMap::operator*(const PolynomialMap& right) const 
{
	PolynomialMap poly;
	for (const auto& term1 : m_Polynomial)
	{
		for (const auto& term2 : right.m_Polynomial)
		{
			int deg = term1.first + term2.first;
			double cof = term1.second * term2.second;
			poly.coff(deg) += cof;
		}
	}
	return poly; // you should return a correct value
}

PolynomialMap& PolynomialMap::operator=(const PolynomialMap& right) 
{
	m_Polynomial = right.m_Polynomial;
	return *this;
}

void PolynomialMap::Print() const 
{
	auto itr = m_Polynomial.begin();
	if (itr == m_Polynomial.end())
	{
		cout << "0" << endl;
		return;
	}
	for (; itr != m_Polynomial.end(); itr++)
	{
		if (itr != m_Polynomial.begin())
		{
			cout << " ";
			if (itr->second > 0)
				cout << "+";
		}
		cout << itr->second;
		if (itr->first > 0)
			cout << "x^" << itr->first;
	}
	cout << endl;
}

bool PolynomialMap::ReadFromFile(const string& file) 
{
    m_Polynomial.clear();
	ifstream fData;
	fData.open(file.c_str());
	if (!fData.is_open())
	{
		cout << "ERROR::PolynomialList::ReadFromFile:" << endl
			<< "\t" << "file [" << file << "] opens failed" << endl;
		return false;
	}

	char ch;
	int n;
	fData >> ch;
	fData >> n;
	for (int i = 0; i < n; i++) 
	{
		int deg;
		double cof;
		fData >> deg;
		fData >> cof;
		coff(deg) = cof;
	}

	fData.close();

	return true; // you should return a correct value
}
