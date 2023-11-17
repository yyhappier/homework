#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<Windows.h>

void COUNTING_SORT(int* A,int* B, int k,int n)
{//A是输入,B存放输出，C提供临时存储空间
    // double time;
    // LARGE_INTEGER start, end, tc;
    // QueryPerformanceFrequency(&tc); //获取机器内部计时器的时钟频率
    // QueryPerformanceCounter(&start);
    int C[k];   // C[i] 保存等于i的元素的个数
    // int *C = (int *)malloc(sizeof(int) * k);
    // QueryPerformanceCounter(&end);
    // time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart; // 利用两次获得的计数之差和时钟频率，计算出事件经历的精确时间
    // printf("\n初始化C[k]耗时:%lf",time);
    // QueryPerformanceCounter(&start);
    int i=0,j=0;
    for(i=0;i<=k;i++)
    {
        C[i]=0;
    }
    for(j=0;j<n;j++)
    {
        C[A[j]]=C[A[j]]+1;
    }
    for(i=1;i<=k;i++)
    {
        C[i]=C[i]+C[i-1];
    }
    for(j=n-1;j>=0;j--)
    {
        B[C[A[j]]-1]=A[j];
        C[A[j]]=C[A[j]]-1;
    }
    // QueryPerformanceCounter(&end);
    // time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart;
    // printf("\n排序耗时:%lf\n",time);
}

// 顺序读取文件中的元素
// int Get_Num(int *A, int N)
// {
//     int i = 0;
//     FILE *fp = fopen("..\\input\\input.txt", "r");
//     if (fp == NULL)
//     {
//         printf("文件读取无效.\n");
//         return -1;
//     }
//     for (i = 0; i < N; i++)
//         fscanf(fp, "%d", &A[i]);

//     fclose(fp);
//     return 0;
// }

//测试
// int main()
// {
//     int test_array1[8] = {0};
//     int test_array2[64] = {0};
//     int result_array1[8]={0};
//     int result_array2[64]={0};
//     Get_Num(test_array1,8);
//     Get_Num(test_array2,64);
//     printf("2^3:");
//     COUNTING_SORT(test_array1, result_array1,pow(2, 15) - 1, 8);
//     for (int i = 0; i < 8; i++)
//     {
//         printf("%d ", result_array1[i]);
//     }
//     printf("\n2^6:");
//     COUNTING_SORT(test_array2, result_array2,pow(2, 15) - 1, 64);
//     for (int i = 0; i < 64; i++)
//     {
//         printf("%d ", result_array2[i]);
//     }
//     return 0;
// }
