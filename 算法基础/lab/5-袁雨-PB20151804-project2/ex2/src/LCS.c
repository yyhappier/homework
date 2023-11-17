#include <stdio.h>
#include <Windows.h>

void FPRINT_LCS(int *b, char *x, int m, int i, int j ,FILE *fp);
void LCS_LENGTH(char *x,char *y,int m,int n,FILE *fp);

int main()
{
    int i = 0,n=5;
    double time;
    char fileName[50];
    LARGE_INTEGER start, end, tc;
    // 打开文件
    FILE *fp0 = fopen("..\\input\\2_2_input.txt", "r");
    FILE *fp2 = fopen( "..\\output\\time.txt", "w");
    if (fp0 == NULL||fp2 == NULL)
    {
        printf("文件读取失败！\n");
        return -1;
    }

   // 读入数据并调用函数
    while(fscanf(fp0,"%d",&n)!=EOF)
    {
        char x[n+1];
        char y[n+1];
        for (i = 1; i <= n ; i++)
        {
            fscanf(fp0,"%c",&x[i]);
            if(x[i]=='\n')
            {
                fscanf (fp0,"%c",&x[i]);
            }
        }
        for (i = 1; i <= n ; i++)
        {
            fscanf(fp0,"%c",&y[i]);
            if(y[i]=='\n')
            {
                fscanf (fp0,"%c",&y[i]);
            }
        }
        sprintf(fileName, "%s%s%d%s", "..\\output\\","\\result_", n, ".txt");
        FILE *fp1 = fopen(fileName, "w");
        if (fp1 == NULL)
        {
            printf("文件读取失败！\n");
            return -1;
        }
        //计时
        QueryPerformanceFrequency(&tc); 
        QueryPerformanceCounter(&start);
        LCS_LENGTH(x, y, n, n,fp1);
        QueryPerformanceCounter(&end);
        time = (end.QuadPart - start.QuadPart) * 1.0 / tc.QuadPart; 
        fprintf(fp2, "%lf\n", time);// 将运行时间写入文件
        fclose(fp1);
    }

    //关闭文件
    fclose(fp0);
    fclose(fp2);

    return 0;
}

// 输出到文件
void FPRINT_LCS(int *b, char *x, int m, int i, int j ,FILE *fp)
{
    if(i==0 || j==0)
    {
        return;
    }
    if(*(b+(i*(m+1)+j))==0)
    {
        FPRINT_LCS(b,x,m,i-1,j-1,fp);
        // printf("%c",x[i]);
        fprintf(fp,"%c",x[i]);
    }
    else if(*(b+(i*(m+1)+j))==1)
    {
        FPRINT_LCS(b,x,m,i-1,j,fp);
    }
    else
    {
        FPRINT_LCS(b,x,m,i,j-1,fp);
    }
}

// 动态规划求解
void LCS_LENGTH(char *x,char *y,int m,int n,FILE *fp)
{
    int b[m+1][n+1],c[m+1][n+1];
    int i=0,j=0;
    for(i=1;i<=m;i++)
    {
        c[i][0]=0;
    }
    for(j=0;j<=n;j++)
    {
        c[0][j]=0;
    }
    for(i=1;i<=m;i++)
    {
        for(j=1;j<=n;j++)
        {
            if(x[i]==y[j])
            {
                c[i][j]=c[i-1][j-1]+1;
                b[i][j]=0;
            }
            else if(c[i-1][j]>=c[i][j-1])
            {
                c[i][j] = c[i-1][j];
                b[i][j]=1;
            }
            else
            {
                c[i][j]=c[i][j-1];
                b[i][j]=-1;
            }
        }
    }
    fprintf(fp,"%d\n",c[m][n]);
    FPRINT_LCS(*b,x,m,m,n,fp);
    fprintf(fp,"\n");
}
