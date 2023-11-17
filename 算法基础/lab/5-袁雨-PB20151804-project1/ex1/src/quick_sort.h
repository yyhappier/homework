#include <stdio.h>
#include <stdlib.h>

// 交换函数
/*
void SWAP(int *a, int *b)
{
    int temp = 0;
    temp = *a;
    *a = *b;
    *b = temp;
}
*/

// 数组的划分
int PARTITION(int *A, int p, int r)
{
    int x = A[r];   // 主元
    int i = p - 1;
    for (int j = p; j <= r - 1; j++)
    {
        if (A[j] <= x)
        {
            i = i + 1;
            SWAP(&A[i], &A[j]);
        }
    }
    SWAP(&A[i + 1], &A[r]);
    return i + 1;
}

// 快速排序
void QUICKSORT(int *A, int p, int r)
{
    int q = 0;
    if (p < r)
    {
        q = PARTITION(A, p, r);
        QUICKSORT(A, p, q - 1);
        QUICKSORT(A, q + 1, r);
    }
}

// 测试
/*
int main()
{
    int test_array[9] = {11, 4, 5, 9, 6, 8, 7, 1, 2};
    QUICKSORT(test_array, 0, 8);
    for (int i = 0; i < 9; i++)
    {
        printf("%d ", test_array[i]);
    }
    return 0;
}
*/