#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <queue>
using namespace std;
#define MAX 100000

typedef struct ANode // 边节点
{
    int key;     // 节点值
    int w;       // 边权值
    int w_;      // 修改后的边权值
    ANode *next; // 指向下一条边的指针
} ANode;

typedef struct Head // 节点信息
{
    Head() { adj = NULL; }
    int d;      // 最短路径估计
    int parent; // 前驱节点
    int h;      // 映射后的值
    ANode *adj; // 邻接点
} Head;

typedef struct VNode // 头节点
{
    int key;    // 节点值
    Head *head; // 节点信息
} VNode;

typedef struct cmp
{
    bool operator()(VNode a, VNode b)
    {
        return a.head->d > b.head->d;
    }
} cmp;

void InitializeSingleSource(Head *G, int N, int s)
{
    for (int i = 0; i < N; i++)
    {
        G[i].d = MAX;
        G[i].parent = -1;
    }
    G[s].d = 0;
}

void KillNegEdges(Head *G, int N, int p)
{

    int i = 0, u = 0, v = 0;
    for (i = 0; i < N; i++)
    {
        for (u = 0; u < N; u++)
        {
            ANode *adj = G[u].adj;
            ANode *headd = adj;
            ANode *parent = adj;
            while (adj)
            {
                v = adj->key;
                if (G[v].d > G[u].d + adj->w)
                {
                    if (p == v)
                    {
                        printf("There is a negative-weight cycle from %d to %d.\n", u, p);
                        if (adj == headd)
                        {
                            G[u].adj = adj->next;
                            delete adj;
                        }
                        else
                        {
                            parent->next = adj->next;
                            delete adj;
                        }
                        // adj->w = MAX;

                        return;
                    }
                    else
                    {
                        G[v].d = G[u].d + adj->w;
                        G[v].parent = u;
                    }
                }
                parent = adj;
                adj = adj->next;
            }
        }
    }
}

bool BellmanFord(Head *G, int N, int s)
{
    InitializeSingleSource(G, N, s);
    int i = 0, u = 0, v = 0;
    for (i = 0; i < N; i++)
    {
        for (u = 0; u < N; u++)
        {
            ANode *adj = G[u].adj;
            while (adj)
            {
                v = adj->key;
                if (G[v].d > G[u].d + adj->w)
                {
                    G[v].d = G[u].d + adj->w;
                    G[v].parent = u;
                }
                adj = adj->next;
            }
        }
    }
    for (u = 0; u < N; u++)
    {
        ANode *adj = G[u].adj;
        while (adj)
        {
            v = adj->key;
            if (G[v].d > G[u].d + adj->w)
            {
                // printf("There is a negative-weight cycle in %d.\n", u);
                KillNegEdges(G, N, u);
                return false;
            }
            adj = adj->next;
        }
    }
    return true;
}

void Dijkstra(Head *G, int N, int s)
{
    InitializeSingleSource(G, N, s);
    int u = 0, v = 0;
    priority_queue<VNode, vector<VNode>, cmp> Q; // 最小优先队列，关键值为d值
    vector<int> S;                               // 已找到最短路径的结点
    VNode temp;
    temp.key = s;
    temp.head = &G[s];
    Q.push(temp);
    while (!Q.empty())
    {
        // EXTRACT-MIN(Q)
        VNode unode = Q.top();
        u = unode.key;
        Q.pop();
        S.push_back(u);
        ANode *adj = G[u].adj;
        while (adj)
        {
            v = adj->key - 1;
            if (G[v].d > G[u].d + adj->w_)
            {
                G[v].d = G[u].d + adj->w_;
                G[v].parent = u + 1;
                VNode temp;
                temp.key = v;
                temp.head = &G[v];
                Q.push(temp);
            }
            adj = adj->next;
        }
    }
}

void PrintPath(Head *G, int u, int v, FILE *output)
{
    if (u != v)
    {
        PrintPath(G, u, G[v].parent, output);
        fprintf(output, ",%d", v);
    }
    else
    {
        fprintf(output, "(%d", u);
    }
}

double Johnson(Head *G, int N, FILE *output)
{
    int u = 0, v = 0;
    for (u = 1; u <= N; u++)
    {
        ANode *temp = new (ANode);
        temp->key = u;
        temp->w = 0;
        temp->next = G[0].adj;
        G[0].adj = temp;
    }
    int delnum = 0;
    while (1)
    {
        if ((BellmanFord(G, N + 1, 0)) == false)
            delnum++;
        else
            break;
    }
    printf("删除了%d条边\n", delnum);
    for (v = 0; v <= N; v++)
        G[v].h = G[v].d;
    for (u = 0; u <= N; u++)
    {
        ANode *temp = G[u].adj;
        while (temp)
        {
            temp->w_ = temp->w + G[u].h - G[temp->key].h;
            temp = temp->next;
        }
    }
    int d[N][N];
    for (u = 0; u < N; u++)
    {
        for (v = 0; v < N; v++)
        {
            d[u][v] = -MAX;
        }
    }
    double time1 = clock();
    for (u = 1; u <= N; u++)
    {
        Dijkstra(&G[1], N, u - 1);
        for (v = 1; v <= N; v++)
        {
            if (v != u)
            {
                if (G[v].d != MAX)
                {
                    d[u - 1][v - 1] = G[v].d + G[v].h - G[u].h;
                    PrintPath(G, u, v, output);
                    fprintf(output, " %d)\n", d[u - 1][v - 1]);
                }
                else
                    fprintf(output, "There is no path from %d to %d.\n", u, v);
            }
        }
    }
    time1 = (clock() - time1) / CLOCKS_PER_SEC;
    return time1;
}

int main()
{
    int i = 0, j = 0, u = 0, v = 0, w = 0, n = 0, vpre = -1;
    int num[4][3] = {{27}, {81}, {243}, {729}};
    char inputName[50];
    char outputName[50];
    double t;
    // 计算边数
    for (i = 0; i < 4; i++)
    {
        num[i][1] = log(num[i][0]) / log(5);
        cout << num[i][1] << ' ';
        num[i][2] = log(num[i][0]) / log(7);
        cout << num[i][2] << endl;
    }
    FILE *ftime1 = fopen("../output/time.txt", "w");
    fclose(ftime1);
    for (i = 0; i < 4; i++)
    {
        for (j = 1; j <= 2; j++)
        {

            sprintf(inputName, "%s%s%d%d%s", "../input/", "input", i + 1, j, ".txt");
            sprintf(outputName, "%s%s%d%d%s", "../output/", "/output", i + 1, j, ".txt");
            FILE *finw = fopen(inputName, "r");
            FILE *foutw = fopen(outputName, "w");
            FILE *ftime = fopen("../output/time.txt", "a");

            Head G[num[i][0] + 1];
            // 随机生成有向图信息
            srand(time(NULL));
            for (u = 1; u <= num[i][0]; u++)
            {
                for (n = 1; n <= num[i][j]; n++)
                {
                    while (1)
                    {
                        v = rand() % num[i][0] + 1; //[1,num[i][0]]
                        if (v != u && v != vpre)    // 排除自环和多重边
                            break;
                    }
                    vpre = v;
                    w = rand() % 61 - 10; //[-10,50]
                    fprintf(finw, "%d %d %d\n", u, v, w);

                    ANode *temp = new (ANode);
                    temp->key = v;
                    temp->w = w;
                    temp->next = G[u].adj;
                    G[u].adj = temp;
                }
            }

            double t = Johnson(G, num[i][0], foutw);

            fprintf(ftime, "%lf\n", t);

            for (u = 0; u <= num[i][0]; u++)
            {
                ANode *temp = G[u].adj;
                ANode *del = G[u].adj;
                while (temp)
                {
                    delete del;
                    temp = temp->next;
                    del = temp;
                }
            }

            fclose(finw);
            fclose(foutw);
            fclose(ftime);
        }
    }
    return 0;
}