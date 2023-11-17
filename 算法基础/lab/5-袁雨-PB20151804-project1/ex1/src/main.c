#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <Windows.h>

// 引入排序算法
#include "heap_sort.h"
#include "quick_sort.h"
#include "merge_sort.h"
#include "counting_sort.h"

#define NUMBER pow(2, 18)
#define MAX pow(2, 15) - 1

// 生成随机数存入文件
void Create_Num()
{
    srand(time(NULL)); //随机数
    FILE *src = fopen("..\\input\\input.txt", "w+");
    for (int i = 0; i < NUMBER; i++)
    {
        fprintf(src, "%d\n", rand()); //写入随机数
    }
    fclose(src);
}

// 顺序读取文件中的元素
int Get_Num(int *A, int N)
{
    int i = 0;
    FILE *fp = fopen("..\\input\\input.txt", "r");
    if (fp == NULL)
    {
        printf("文件读取无效.\n");
        return -1;
    }
    for (i = 0; i < N; i++)
        fscanf(fp, "%d", &A[i]);

    fclose(fp);
    return 0;
}

// 保存排序结果到输出文件
int Save_Result(int *A, double time, int N, char *sort)
{
    int i = 0;
    char fileName[50];
    char timeName[50];
    sprintf(fileName, "%s%s%s%d%s", "..\\output\\", sort, "\\result_", N, ".txt");
    sprintf(timeName, "%s%s%s", "..\\output\\", sort, "\\time.txt");
    // printf("%s", fileName);
    FILE *fp1 = fopen(fileName, "w");
    if (fp1 == NULL)
    {
        printf("文件读取无效.\n");
        return -1;
    }
    FILE *fp2 = fopen(timeName, "a");
    if (fp2 == NULL)
    {
        printf("文件读取无效.\n");
        return -1;
    }
    // 写入排序结果
    for (i = 0; i < pow(2, N); i++)
    {
        fprintf(fp1, "%d\n", A[i]);
    }
    // 写入运行时间
    fprintf(fp2, "%lf\n", time);
    fclose(fp1);
    fclose(fp2);
    return 0;
}

int main()
{
    Create_Num();
    int i = 0, j = 0;
    int n = 0;
    char fileName[50];
    double time;
    LARGE_INTEGER start, end, tc;
    for (n = 3; n <= 18; n += 3)    // 遍历每个数据规模
    {
        int N = pow(2, n);
        // 使用malloc 避免使用数组空间不够
        int *A = (int *)malloc(sizeof(int) * N); // 待排序数据
        int *B = (int *)malloc(sizeof(int) * N); // 存放计数排序的输出
        int *C = (int *)malloc(sizeof(int) * N); // 存放A数组的拷贝
        Get_Num(A, N);
        memcpy(C, A, N*sizeof(int));
        for (j = 0; j <= 3; j++)    // 遍历每个排序算法
        {
            switch (j)
            {
            case 0: // 堆排序
            {
                HEAP HEAPA = {
                    N,
                    N,
                    A};
                QueryPerformanceFrequency(&tc); //获取机器内部计时器的时钟频率
                QueryPerformanceCounter(&start);
                HEAP_SORT(&HEAPA);
                QueryPerformanceCounter(&end);
                time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart; // 利用两次获得的计数之差和时钟频率，计算出事件经历的精确时间
                Save_Result(A, time, n, "heap_sort");
                memcpy(A, C, N*sizeof(int));
                break;
            }
            case 1: // 快速排序
            {
                Get_Num(A, N);
                QueryPerformanceFrequency(&tc);
                QueryPerformanceCounter(&start);
                QUICKSORT(A, 0, N - 1);
                QueryPerformanceCounter(&end);
                time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart;
                Save_Result(A, time, n, "quick_sort");
                memcpy(A, C, N*sizeof(int));
                break;
            }
            case 2: //归并排序
            {
                QueryPerformanceFrequency(&tc);
                QueryPerformanceCounter(&start);
                MERGE_SORT(A, 0, N-1);
                QueryPerformanceCounter(&end);
                time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart;
                Save_Result(A, time, n, "merge_sort");
                memcpy(A, C, N*sizeof(int));
                break;
            }
            case 3: // 计数排序
            {
                QueryPerformanceFrequency(&tc);
                QueryPerformanceCounter(&start);
                COUNTING_SORT(A, B, MAX, N);
                QueryPerformanceCounter(&end);
                time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart;
                Save_Result(B, time, n, "counting_sort");
                break;
            }
            default:
                break;
            }
        }
        free(A);
        free(B);
        free(C);

    }
    
    return 0;
}
