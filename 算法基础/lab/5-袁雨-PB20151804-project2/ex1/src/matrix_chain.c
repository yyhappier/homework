#include <stdio.h>
#include <Windows.h>
#define max_int 9223372036854775807

void FPRINT_OPTIMAL_PARENS(long long *s, int n, int i, int j, FILE *fp);
void MATRIX_CHAIN_ORDER(long long *p, int n, FILE *fp);

int main()
{
    int i = 0, n = 5;
    double time;
    LARGE_INTEGER start, end, tc;
    // 打开文件
    FILE *fp0 = fopen("..\\input\\2_1_input.txt", "r");
    FILE *fp1 = fopen("..\\output\\result.txt", "w");
    FILE *fp2 = fopen("..\\output\\time.txt", "w");
    if (fp0 == NULL || fp1 == NULL || fp2 == NULL)
    {
        printf("文件读取失败！\n");
        return -1;
    }
    
    // 读入数据并调用函数
    while (fscanf(fp0, "%d", &n) != EOF)
    {
        long long p[n + 1];
        for (i = 0; i <= n; i++)
        {
            fscanf(fp0, "%lld", &p[i]);
        }
        // 计时
        QueryPerformanceFrequency(&tc);
        QueryPerformanceCounter(&start);
        MATRIX_CHAIN_ORDER(p, n, fp1);
        QueryPerformanceCounter(&end);
        time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart;
        fprintf(fp2, "%lf\n", time); // 将运行时间写入文件
    }

    // 关闭文件
    fclose(fp0);
    fclose(fp1);
    fclose(fp2);

    return 0;
}

// 输出到文件
void FPRINT_OPTIMAL_PARENS(long long *s, int n, int i, int j, FILE *fp)
{
    if (i == j)
    {
        // printf("A%d", i);
        fprintf(fp, "A%d", i);
    }
    else
    {
        fprintf(fp, "(");
        FPRINT_OPTIMAL_PARENS(s, n, i, *(s + (i * (n + 1) + j)), fp);
        FPRINT_OPTIMAL_PARENS(s, n, *(s + (i * (n + 1) + j)) + 1, j, fp);
        fprintf(fp, ")");
    }
}

// 动态规划求解
void MATRIX_CHAIN_ORDER(long long *p, int n, FILE *fp)
{
    long long m[n + 1][n + 1], s[n][n + 1]; // m[i][j]：子问题最优解的代价 s[i][j]：最优括号化方案的分割点位置
    int i = 0, j = 0, l = 0, k = 0;
    long long q = 0;
    for (i = 1; i <= n; i++)
    {
        m[i][i] = 0;
    }
    for (l = 1; l <= n - 1; l++)
    {
        for (i = 1; i <= n - l; i++)
        {
            j = i + l;
            m[i][j] = max_int;
            for (k = i; k <= j - 1; k++)
            {
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < m[i][j])
                {
                    m[i][j] = q;
                    s[i][j] = k;
                }
            }
        }
    }
    // if(n==5)
    // {
    //     printf("n=5:\n");
    //     printf("\nm\n");
    //     printf("i\\j\t\t5\t\t\t\t4\t\t\t\t3\t\t\t\t2\t\t\t\t1 \n");
    //     for(i = 5; i >= 1; i--)
    //     {
    //         printf("%d\t",i);
    //         for(int j = 5; j >= i; j--)
    //         {
    //             printf("\t%lld",m[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\ns\n");
    //     printf("i\\j\t5\t4\t3\t2 \n");
    //     for(int i = 4; i >= 1; i--)
    //     {
    //         printf("%d\t",i);
    //         for(int j = 5; j >=i+1; j--)
    //         {
    //             printf("%lld\t",s[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }
    fprintf(fp, "%lld\n", m[1][n]);
    FPRINT_OPTIMAL_PARENS(*s, n, 1, n, fp);
    fprintf(fp, "\n");
}
