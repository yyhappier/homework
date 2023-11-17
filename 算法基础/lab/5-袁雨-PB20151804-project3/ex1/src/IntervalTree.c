#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

#define RED 0
#define BLACK 1

typedef struct Interval
{
    int low;
    int high;
} Interval;

typedef struct Node
{
    int color;
    struct Node *parent;
    struct Node *left;
    struct Node *right;
    Interval interval;
    int max;
} Node;

typedef struct RBTree
{
    Node *root;
    Node *NIL;
} RBTree;

int MAX(int high, int leftMax, int rightMax);         // 求三者中的最大值
Node *IntervalMin(RBTree *T, Node *x);                // 求子树中具有最小关键字的结点
RBTree *IntervalCreateTree();                         // 创建区间树
Node *IntervalCreateNode(int low, int high);          // 创建区间为[low,high]的结点
Node *IntervalFindNode(RBTree *T, int num);           // 寻找关键字为num的结点
int OverLap(Node *x, Node *y);                        // 判断x与y的区间是否重叠
Node *IntervalSearch(RBTree *T, Node *i);             // 查找与区间i重叠的结点
void LeftRotate(RBTree *T, Node *x);                  // 左旋
void RightRotate(RBTree *T, Node *x);                 // 右旋
void IntervalInsertFixUp(RBTree *T, Node *z);         // 插入结点后保持红黑性质
void IntervalInsert(RBTree *T, Node *z);              // 插入结点
void IntervalTransplant(RBTree *T, Node *u, Node *v); // 删除结点的子过程
void IntervalDeleteFixUp(RBTree *T, Node *x);         // 删除结点
void IntervalDelete(RBTree *T, Node *z);              // 删除结点后保持红黑性质
void InOrder(FILE *fp, RBTree *T, Node *x);           // 中序遍历

int main()
{
    FILE *fp_in = fopen("..\\input\\input.txt", "w+");
    FILE *fp_out1 = fopen("..\\output\\inorder.txt", "w");
    FILE *fp_out2 = fopen("..\\output\\delete_data.txt", "w");
    FILE *fp_out3 = fopen("..\\output\\search.txt", "w");

    int i = 0, choice = 0, temp = 0, templow = 0, temphigh = 0, delIndex = 0;
    int num[51] = {0}, low[30] = {0}, high[30] = {0}, searchlow[3] = {0}, searchhigh[3] = {0};
    Node *x, *delNode, *searchNode;

    // 随机生成30个左端点互异的自然数区间
    srand((unsigned)time(NULL));
    for (i = 0; i < 30; i++)
    {
        choice = rand() % 2;
        if (choice == 0)
        { //[0,25]
            low[i] = rand() % 26;
            while (num[low[i]] == 1)
            {
                low[i] = rand() % 26;
                // printf("num[%d]:%d ", low[i], num[low[i]]);
            }
            high[i] = low[i] + rand() % (25 - low[i] + 1);
        }
        else
        { //[30,50]
            low[i] = 30 + rand() % 21;
            while (num[low[i]] == 1)
            {
                low[i] = 30 + rand() % 21;
                // printf("num[%d]:%d ", low[i], num[low[i]]);
            }
            high[i] = low[i] + rand() % (50 - low[i] + 1);
        }

        fprintf(fp_in, "%d %d\n", low[i], high[i]);
        num[low[i]] = 1;
    }

    rewind(fp_in);

    // 构建红黑树
    RBTree *T = IntervalCreateTree();
    // 依次插入30个节点
    for (i = 0; i < 30; i++)
    {
        fscanf(fp_in, "%d %d\n", &templow, &temphigh);
        x = IntervalCreateNode(templow, temphigh);
        IntervalInsert(T, x);
    }
    fclose(fp_in);

    // 中序遍历
    InOrder(fp_out1, T, T->root);
    fclose(fp_out1);

    // 随机选择其中三个区间进行删除
    for (i = 0; i < 3; i++)
    {
        delIndex = i * 10 + rand() % 11;
        delNode = IntervalFindNode(T, low[delIndex]);
        // 输出删除的数据
        fprintf(fp_out2, "%d %d %d\n", delNode->interval.low, delNode->interval.high, delNode->max);
        IntervalDelete(T, delNode);
    }
    // 删除完成后的中序遍历序列
    InOrder(fp_out2, T, T->root);
    fclose(fp_out2);

    // 随机生成三个区间进行搜索
    //(25,30)
    searchlow[0] = 26 + rand() % (29 - 26 + 1);
    searchhigh[0] = searchlow[0] + rand() % (29 - searchlow[0] + 1);
    //[0,25]
    searchlow[1] = rand() % (25 + 1);
    searchhigh[1] = searchlow[1] + rand() % (25 - searchlow[1] + 1);
    //[30,50]
    searchlow[2] = 30 + rand() % (50 - 30 + 1);
    searchhigh[2] = searchlow[2] + rand() % (50 - searchlow[2] + 1);
    for (i = 0; i < 3; i++)
    {
        x = IntervalCreateNode(searchlow[i], searchhigh[i]);
        searchNode = IntervalSearch(T, x);
        fprintf(fp_out3, "%d %d ", searchlow[i], searchhigh[i]);
        if (searchNode == T->NIL)
            fprintf(fp_out3, "");
        else
            fprintf(fp_out3, "%d %d", searchNode->interval.low, searchNode->interval.high);
        fprintf(fp_out3, "\n");
    }
    fclose(fp_out3);
}

int MAX(int high, int leftMax, int rightMax)
{
    int temp = (leftMax >= rightMax) ? leftMax : rightMax;
    return (high >= temp) ? high : temp;
}

Node *IntervalMin(RBTree *T, Node *x)
{
    Node *y = x;
    while (y->left != T->NIL)
    {
        y = y->left;
    }
    return y;
}

RBTree *IntervalCreateTree()
{
    RBTree *T = (RBTree *)malloc(sizeof(RBTree));
    T->NIL = (Node *)malloc(sizeof(Node));
    T->NIL->left = T->NIL;
    T->NIL->right = T->NIL;
    T->NIL->parent = T->NIL;
    T->NIL->color = BLACK;
    T->NIL->interval.low = -1;
    T->NIL->interval.high = -1;
    T->NIL->max = 0;
    T->root = T->NIL;
    return T;
}

Node *IntervalCreateNode(int low, int high)
{
    Node *x = (Node *)malloc(sizeof(Node));
    x->interval.low = low;
    x->interval.high = high;
    x->max = high;
    return x;
}

Node *IntervalFindNode(RBTree *T, int num)
{
    Node *x = T->root;
    while (x != T->NIL)
    {
        if (x->interval.low < num)
            x = x->right;
        else if (x->interval.low == num)
            break;
        else
            x = x->left;
    }
    return x;
}

int OverLap(Node *x, Node *y)
{
    if (x->interval.low > y->interval.high || x->interval.high < y->interval.low)
        return 0;
    else
        return 1;
}

Node *IntervalSearch(RBTree *T, Node *i)
{
    Node *x = T->root;
    while ((x != T->NIL) && (!OverLap(i, x)))
    {
        if (x->left != T->NIL && x->left->max >= i->interval.low)
            x = x->left;
        else
            x = x->right;
    }

    return x;
}

void LeftRotate(RBTree *T, Node *x)
{
    if (x->right != T->NIL)
    {
        Node *y = x->right;
        x->right = y->left;
        if (y->left != T->NIL)
            y->left->parent = x;

        y->parent = x->parent;
        if (x->parent == T->NIL)
            T->root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;

        y->left = x;
        x->parent = y;
        x->max = MAX(x->interval.high, x->left->max, x->right->max);
        y->max = MAX(x->interval.high, y->left->max, y->right->max);
    }
    else
    {
        printf("ERROR:NO RIGHT CHILD,CANNOT DO LEFT-ROTATE!\n");
    }
}

void RightRotate(RBTree *T, Node *x)
{
    if (x->left != T->NIL)
    {
        Node *y = x->left;
        x->left = y->right;
        if (y->right != T->NIL)
            y->right->parent = x;

        y->parent = x->parent;

        if (x->parent == T->NIL)
            T->root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;

        y->right = x;
        x->parent = y;

        x->max = MAX(x->interval.high, x->left->max, x->right->max);
        y->max = MAX(y->interval.high, y->left->max, y->right->max);
    }
    else
    {
        printf("ERROR:NO LEFT CHILD,CANNOT DO RIGHT-ROTATE!\n");
    }
}

void IntervalInsertFixUp(RBTree *T, Node *z)
{
    Node *y;
    while (z->parent->color == RED)
    {
        if (z->parent == z->parent->parent->left)
        {
            y = z->parent->parent->right;
            if (y->color == RED)
            {
                z->parent->color = BLACK;
                y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            }
            else
            {
                if (z == z->parent->right)
                {
                    z = z->parent;
                    LeftRotate(T, z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                RightRotate(T, z->parent->parent);
            }
        }
        else
        { // same as then clause with "right" and "left" exchanged
            y = z->parent->parent->left;
            if (y->color == RED)
            {
                z->parent->color = BLACK;
                y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            }
            else
            {
                if (z == z->parent->left)
                {
                    z = z->parent;
                    RightRotate(T, z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                LeftRotate(T, z->parent->parent);
            }
        }
    }
    T->root->color = BLACK;
}

void IntervalInsert(RBTree *T, Node *z)
{
    Node *y = T->NIL;
    Node *x = T->root;
    while (x != T->NIL)
    {
        y = x;
        if (y->max < z->interval.high)
            y->max = z->interval.high;

        if (z->interval.low < x->interval.low)
            x = x->left;
        else
            x = x->right;
    }
    z->parent = y;
    if (y == T->NIL)
        T->root = z;
    else if (z->interval.low < y->interval.low)
        y->left = z;
    else
        y->right = z;

    z->left = T->NIL;
    z->right = T->NIL;
    z->color = RED;

    z->max = z->interval.high;
    Node *temp = z->parent;
    while (temp != T->NIL)
    {
        temp->max = MAX(temp->interval.high, temp->left->max, temp->right->max);
        temp = temp->parent;
    }
    IntervalInsertFixUp(T, z);
}

void IntervalTransplant(RBTree *T, Node *u, Node *v)
{
    if (u->parent == T->NIL)
        T->root = v;
    else if (u == u->parent->left)
        u->parent->left = v;
    else
        u->parent->right = v;

    v->parent = u->parent;
}

void IntervalDeleteFixUp(RBTree *T, Node *x)
{
    Node *w;
    while (x != T->root && x->color == BLACK)
    {
        if (x == x->parent->left)
        {
            w = x->parent->right;
            if (w->color == RED)
            {
                w->color = BLACK;
                x->parent->color = RED;
                LeftRotate(T, x->parent);
                w = x->parent->right;
            }
            if (w->left->color == BLACK && w->right->color == BLACK)
            {
                w->color = RED;
                x = x->parent;
            }
            else
            {
                if (w->right->color == BLACK)
                {
                    w->left->color = BLACK;
                    w->color = RED;
                    RightRotate(T, w);
                    w = x->parent->right;
                }
                w->color = x->parent->color;
                x->parent->color = BLACK;
                w->right->color = BLACK;
                LeftRotate(T, x->parent);
                x = T->root;
            }
        }
        else
        { // same as then clause with"right" and "left" exchanged
            w = x->parent->left;
            if (w->color == RED)
            {
                w->color = BLACK;
                x->parent->color = RED;
                RightRotate(T, x->parent);
                w = x->parent->left;
            }
            if (w->left->color == BLACK && w->right->color == BLACK)
            {
                w->color = RED;
                x = x->parent;
            }
            else
            {
                if (w->left->color == BLACK)
                {
                    w->right->color = BLACK;
                    w->color = RED;
                    LeftRotate(T, w);
                    w = x->parent->left;
                }
                w->color = x->parent->color;
                x->parent->color = BLACK;
                w->left->color = BLACK;
                RightRotate(T, x->parent);
                x = T->root;
            }
        }
    }
    x->color = BLACK;
}

void IntervalDelete(RBTree *T, Node *z)
{
    z->max = 0;
    Node *temp = z;
    while (temp != T->NIL)
    {
        temp->max = MAX(temp->interval.high, temp->left->max, temp->right->max);
        temp = temp->parent;
    }
    Node *y = z;
    int y_original_color = y->color;
    Node *x;
    if (z->left == T->NIL)
    {
        x = z->right;
        IntervalTransplant(T, z, z->left);
    }
    else if (z->right == T->NIL)
    {
        x = z->left;
        IntervalTransplant(T, z, z->left);
    }
    else
    {
        y = IntervalMin(T, z->right);
        y_original_color = y->color;
        x = y->right;
        if (y->parent == z)
        {
            x->parent = y;
        }
        else
        {
            IntervalTransplant(T, y, y->right);
            y->right = z->right;
            y->right->parent = y;
        }
        IntervalTransplant(T, z, y);
        y->left = z->left;
        y->left->parent = y;
        y->color = z->color;
    }
    if (y_original_color == BLACK)
    {
        IntervalDeleteFixUp(T, x);
    }
    free(z);
}

void InOrder(FILE *fp, RBTree *T, Node *x)
{
    if (x != T->NIL)
    {
        InOrder(fp, T, x->left);
        fprintf(fp, "%d %d %d\n", x->interval.low, x->interval.high, x->max);
        InOrder(fp, T, x->right);
    }
}
