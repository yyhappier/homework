#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// 合并两个已排序序列
void MERGE(int *A, int p, int q, int r)
{
    int n1 = q - p + 1;
    int n2 = r - q;
    // int *L = (int *)malloc(sizeof(int) * (n1+1));
    // int *R = (int *)malloc(sizeof(int) * (n2+1));
    int L[n1 + 1], R[n2 + 1];
    int i = 0, j = 0, k = 0;
    for (i = 0; i < n1; i++)
    {
        L[i] = A[p + i];
    }
    for (j = 0; j < n2; j++)
    {
        R[j] = A[q + j + 1];
    }
    L[n1] = INT_MAX;    // 增加监视哨，减少比较次数
    R[n2] = INT_MAX;    // 哨兵
    i = 0;
    j = 0;
    for (k = p; k <= r; k++)
    {
        if (L[i] <= R[j])
        {
            A[k] = L[i];
            i = i + 1;
        }
        else
        {
            A[k] = R[j];
            j = j + 1;
        }
    }
    // free(L);
    // free(R);
}

// 归并排序
void MERGE_SORT(int *A, int p, int r)
{
    if (p < r)
    {
        int q = (p + r) / 2;
        MERGE_SORT(A, p, q);
        MERGE_SORT(A, q + 1, r);
        MERGE(A, p, q, r);
    }
}

// int main()
// {
//     int test_array[16] = {19, 4, 5, 9, 6, 8, 7, 1, 2, 13, 15, 16, 17, 10, 33, 24};
//     MERGE_SORT(test_array, 0, 15);
//     for (int i = 0; i < 16; i++)
//     {
//         printf("%d ", test_array[i]);
//     }
//     return 0;
// }