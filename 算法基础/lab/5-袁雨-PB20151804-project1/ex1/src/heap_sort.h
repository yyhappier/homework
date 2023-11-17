#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 堆的结构
typedef struct HEAP
{
    int heap_size;  //堆的大小
    int length; // 数组的大小
    int *array; // 参与排序的数组
} HEAP;

// 父节点
int PARENT(int i)
{
    return (i - 1) / 2;
}

// 左孩子
int LEFT(int i)
{
    return 2 * i + 1;
}

// 右孩子
int RIGHT(int i)
{
    return 2 * i + 2;
}

// 交换函数
void SWAP(int *a, int *b)
{
    int temp = 0;
    temp = *a;
    *a = *b;
    *b = temp;
}

// 维护最大堆
// 让A[i]的值在最大堆中“逐级下降”
void MAX_HEAPIFY(HEAP *A, int i)
{
    int l = LEFT(i);
    int r = RIGHT(i);
    int largest = 0;
    if (l < A->heap_size && A->array[l] > A->array[i])
    {
        largest = l;
    }
    else
    {
        largest = i;
    }
    if (r < A->heap_size && A->array[r] > A->array[largest])
    {
        largest = r;
    }
    if (largest != i)
    {
        SWAP(&A->array[i], &A->array[largest]);
        MAX_HEAPIFY(A, largest);
    }
}

// 建堆
void BUILD_MAX_HEAP(HEAP *A)
{
    A->heap_size = A->length;
    for (int i = A->length / 2 - 1; i >= 0; i--)
    {
        MAX_HEAPIFY(A, i);
    }
}

// 堆排序
int *HEAP_SORT(HEAP *A)
{
    BUILD_MAX_HEAP(A);
    for (int i = A->length - 1; i >= 1; i--)
    {
        SWAP(&A->array[0], &A->array[i]);
        A->heap_size = A->heap_size - 1;
        MAX_HEAPIFY(A, 0);
    }
    return A->array;
}

// 测试
// int main()
// {
//     int test_array[9] = {11, 4, 5, 9, 6, 8, 7, 1, 2};
//     int array_size = 9;
//     HEAP A = {
//         9,
//         array_size,
//         test_array};
//     for (int i = 0; i < array_size; i++)
//     {
//         printf("%d ", test_array[i]);
//     }
//     printf("\n");
//     int *temp = HEAP_SORT(&A);
//     for (int i = 0; i < array_size; i++)
//     {
//         printf("%d ", temp[i]);
//     }
//     return 0;
// }