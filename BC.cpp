#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#endif
#include <vector>
#include <stack>
#include <queue>
using namespace std;
#include "headers.h"
#define INFINITE 1000000000
// #define DEBUGx


typedef struct q_struct{
    int nodeID;
    bool *traverse_S;

}Q_struct;


#pragma region DefineLabel
// #define DEBUG
// #define CheckDistAns
// #define CheckCC_Ans
#pragma endregion //DefineLabel

#pragma region globalVar
int tempSourceID        = 1;
int CheckedNodeCount    = 0;

double forward_Time= 0;
double backward_Time= 0;
double total_time= 0;

double multi_forward_Time= 0;
double multi_backward_Time= 0;
double multi_total_time= 0;


#pragma endregion //globalVar

double time_start                       = 0;
double time_end                         = 0;
double time1                            = 0;
double time2                            = 0;

double multi_time_start                 = 0;
double multi_time_end                   = 0;
double multi_time1                      = 0;
double multi_time2                      = 0;

inline void resetQueue(struct qQueue* _Q){
    _Q->front   = 0;
    _Q->rear    = -1;
    //Q->size如果不變，就不須memcpy
}

void brandes_ORIGIN(const CSR& csr, int V, vector<float>& BC) {
    time_start = seconds();
    struct qQueue* current_queue = InitqQueue();
    struct qQueue* next_queue = InitqQueue();
    qInitResize(current_queue, csr.csrVSize);
    qInitResize(next_queue, csr.csrVSize);

    // 分配記憶體給 sigma, dist, delta 和 二維的 S
    int** S = (int**)malloc(V * sizeof(int*));     // S 是二維陣列（堆疊）
    int* sigma = (int*)malloc(V * sizeof(int));    // sigma 是一維陣列
    int* dist = (int*)malloc(V * sizeof(int));     // dist 是一維陣列
    double* delta = (double*)malloc(V * sizeof(double)); // delta 是一維陣列
    int* S_size = (int*)malloc(V * sizeof(int));   // S_size 紀錄每層的大小

    for (int i = 0; i < V; i++) {
        S[i] = (int*)malloc(V * sizeof(int));      // 每層的堆疊大小 V，根據需要調整
        S_size[i] = 0;                             // 初始化每層的大小
    }

    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        // 初始化每次源節點的變量
        for (int i = 0; i < V; i++) {
            sigma[i] = 0;
            dist[i] = -1;
            delta[i] = 0.0;
            S_size[i] = 0; // 重置每層大小
        }

        sigma[s] = 1;
        dist[s] = 0;

        resetQueue(current_queue);
        qPushBack(current_queue, s);

        time1 = seconds();
        // BFS前進階段：frontier-based BFS
        // BFS前進階段：frontier-based BFS
        while (!qIsEmpty(current_queue) || !qIsEmpty(next_queue)) {
            while (!qIsEmpty(current_queue)) {
                int v = qPopFront(current_queue);
                S[dist[v]][S_size[dist[v]]++] = v;  // 將節點 v 放入其層級

                // 遍歷CSR格式中的鄰接節點
                for (int i = csr.csrV[v]; i < csr.csrV[v + 1]; ++i) {
                    int w = csr.csrE[i];
                    // 如果 w 尚未訪問，更新距離並加入到 next_queue
                    if (dist[w] < 0) {
                        dist[w] = dist[v] + 1;
                        qPushBack(next_queue, w);
                    }

                    // 當找到最短路徑時
                    if (dist[w] == dist[v] + 1) {
                        sigma[w] += sigma[v];
                    }
                }
            }
            // 交替隊列
            std::swap(current_queue, next_queue);
        }
        time2 = seconds();
        forward_Time += (time2 - time1);

        time1 = seconds();
        // 反向計算BC值
        for (int d = V - 1; d >= 0; --d) {  // 從最遠層級開始反向計算
            for (int i = 0; i < S_size[d]; ++i) {
                int w = S[d][i];
                for (int j = csr.csrV[w]; j < csr.csrV[w + 1]; ++j) {
                    int v = csr.csrE[j];
                    if (dist[v] == dist[w] - 1) {
                        delta[v] += (sigma[v] / double(sigma[w])) * (1.0 + delta[w]);
                    }
                }
                if (w != s) {
                    BC[w] += delta[w];
                }
            }
        }
        time2 = seconds();
        backward_Time += (time2 - time1);
    }

    time_end = seconds();
    total_time = (time_end - time_start);
}



void Seq_multi_source_brandes(const CSR& csr,int max_multi,vector<float> &BC) {
    //record that nodes which haven't been source yet
    multi_time_start = seconds();

    bool* nodeDone = (bool*)calloc(csr.csrVSize,sizeof(bool));
    int* map_S = (int*)malloc(sizeof(int) * max_multi);   //多點Source的代表 EX: 213 分別代表source( [0]=2 [1]=1 [2]=3 )
    
    int v_size=csr.csrVSize;
    size_t multi_size= (v_size) * max_multi;
    int*  s_size = (int*)calloc(v_size,sizeof(int));
    int* dist_MULTI = (int*)malloc(sizeof(int) * multi_size);
    float* sigma_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* delta_MULTI = (float*)malloc(sizeof(float) * multi_size);
    bool* visited_MULTI = (bool*)calloc( multi_size , sizeof(bool)); //記錄否有點被每個source記錄
    Q_struct** s = (Q_struct**)malloc(v_size * sizeof(Q_struct*));
    Q_struct* f1 = (Q_struct*)malloc(v_size * sizeof(Q_struct));
    Q_struct* f2 = (Q_struct*)malloc(v_size * sizeof(Q_struct));

    for (int sourceID = csr.startNodeID; sourceID <= csr.endNodeID; ++sourceID) {
        
        multi_time1 = seconds();

        if(nodeDone[sourceID])continue;
        
        nodeDone[sourceID] = true;
        int mappingCount = 0;
        map_S[mappingCount++] = sourceID;

        //找其他 Source
        for(int neighborIndex = csr.csrV[sourceID] ; neighborIndex < csr.csrV[sourceID + 1] && mappingCount < max_multi; neighborIndex ++){
            int neighborNodeID = csr.csrE[neighborIndex];
            //這個鄰居沒當過source才紀錄
            if(!nodeDone[neighborNodeID]){
                map_S[mappingCount++] = neighborNodeID;
                nodeDone[neighborNodeID] = true;
            }
        }
        // for(auto i=0;i<mappingCount;i++){
        //     printf("map_S[%d]: %d\n",i,map_S[i]);
        // }

        // printf("mappingCount: %d\n",mappingCount);
        memset(s_size, 0, sizeof(int) * v_size);
        // memset(dist_MULTI, INFINITE, sizeof(int) * multi_size);
        memset(sigma_MULTI, 0.0, sizeof(float) * multi_size);
        memset(delta_MULTI, 0.0, sizeof(float) * multi_size);
        memset(visited_MULTI, false, sizeof(bool) * multi_size);
        memset(f1, 0, sizeof(Q_struct) * v_size);
        memset(f2, 0, sizeof(Q_struct) * v_size);
        
        //初始值
        for(auto i=0;i<multi_size;i++){
            dist_MULTI[i]  = INFINITE;
        }

        // 10/9
        //第一版
        //每個source的鄰居開始傳播
        //第二版
        //把source之間變成subgraph，讓source間都有彼此的路徑數和距離
        //再傳到非source的點，同步更新多個Source的路徑與數量，並檢查是否正確
        
        //first version
        int f1_indicator=0;
        int f2_indicator=0;
        int s_indicator=0;

        for(auto i=0;i<mappingCount;i++){
            int pos=mappingCount*map_S[i]+i;
            sigma_MULTI[pos] = 1; //初始路徑為1
            dist_MULTI[pos] = 0; //source距離自己為0
            bool* traverse_S = (bool*)calloc(mappingCount, sizeof(bool));
            traverse_S[i]=true;
            f1[f1_indicator++] = {map_S[i],traverse_S};
            visited_MULTI[pos]=true; //初始source已拜訪的點
        }

        #ifdef DEBUGx
            printf("sigma_MULTI:  ");
            for(auto i=0;i<multi_size;i++){
                printf("%d ",sigma_MULTI[i]);
            }
            printf("\n");
            printf("dist_MULTI: ");
            for(auto i=0;i<multi_size;i++){
                printf("%d ",dist_MULTI[i]);
            }
            printf("\n");
        #endif


        int level = 0;
        Q_struct* currentQueue;
        Q_struct* nextQueue;
        while(f1_indicator > 0){ //類似queue是否為empty
            // printf("level: %d\n",level);
            currentQueue = (level % 2 == 0) ? f1 : f2;
            nextQueue = (level % 2 == 0) ? (f2 = (Q_struct*)calloc(csr.csrVSize, sizeof(Q_struct))) : 
                                                    (f1 = (Q_struct*)calloc(csr.csrVSize, sizeof(Q_struct)));

            
            // int queue_size=f1_indicator-1;
            // s[s_indicator] = (Q_struct*)malloc(f1_indicator * sizeof(Q_struct));
            
            // printf("currentQueue size: %d\n",f1_indicator);
            // for(auto i=0;i<f1_indicator;i++){
            //     printf("%d ->",currentQueue[i].nodeID);
            //     for(auto j=0;j<mappingCount;j++)
            //         printf("%d ",currentQueue[i].traverse_S[j]);
            //     printf("\n");
            // }
            // printf("\n");

            s_size[s_indicator]=f1_indicator;
            s[s_indicator++] = currentQueue;


            for(auto i=0;i<f1_indicator;i++){ //這層level所需要traverse的點
                int v=currentQueue[i].nodeID;//nodeID
                // printf("visited node: %d\n",v);

                for(int neighborIndex=csr.csrV[v];neighborIndex<csr.csrV[v+1];neighborIndex++){ //看該點鄰居
                    int neighborNodeID = csr.csrE[neighborIndex];
                    // printf("visited neighbor: %d\n",neighborNodeID);

                    bool* traverse_S = (bool*)calloc(mappingCount, sizeof(bool));//紀錄該點鄰居被幾個source看到
                    for(auto multi_node=0;multi_node<mappingCount;multi_node++){ //要看multi-source的距離以及路徑數
                        int position_n=mappingCount*neighborNodeID+multi_node;
                        int position = mappingCount*v+multi_node;
                        
                        if(currentQueue[i].traverse_S[multi_node]){
                            // 如果找到一條相同長度的最短路徑，累加sigma_MULTI
                            if (dist_MULTI[position_n] == dist_MULTI[position] + 1) {
                                    sigma_MULTI[position_n] += sigma_MULTI[position];
                            } 
                            // 如果找到新的更短路徑，更新距離和sigma_MULTI
                            else if (dist_MULTI[position_n] > dist_MULTI[position] + 1) {
                                dist_MULTI[position_n] = dist_MULTI[position] + 1;
                                sigma_MULTI[position_n] = sigma_MULTI[position]; // 設定為目前的路徑數，重新計算
                                traverse_S[multi_node]=true;

                                bool check_queue=false;
                                for(int find=0;find<f2_indicator;find++){
                                    if(nextQueue[find].nodeID==neighborNodeID){
                                        nextQueue[find].traverse_S[multi_node]=true;
                                        check_queue=true;
                                        // nextQueue[f2_indicator++] = {neighborNodeID,traverse_S};
                                    }
                                }
                                if(!check_queue){
                                    nextQueue[f2_indicator++] = {neighborNodeID,traverse_S};
                                }

                            }
                        }
  
                    }
                   
                }
            }
            
            f1_indicator = f2_indicator;
            f2_indicator = 0;
            level++;

        }
        multi_time2 = seconds();
        multi_forward_Time +=(multi_time2-multi_time1);
        multi_time1 = seconds();
        //backward sum of BC
        for(int layer=s_indicator-1;layer>0;layer--){ //backward層數(不含source層)
            // printf("\nlayer:  %d\n",layer);

            // printf("s_size: %d\n",s_size[layer]);
            //     for(int x=0;x<s_size[layer];x++)
            //         printf("%d ",s[layer][x].nodeID);
            // printf("\n");

            for(int size=0;size<s_size[layer];size++){ //每一層的點所需累加
                int v = s[layer][size].nodeID;
                // printf("v:  %d\n",v);
                // printf("traverse_S: \n");
                // for(int x=0;x<mappingCount;x++)
                //     printf("%d ",s[layer][size].traverse_S[x]);
                // printf("\n");

                for(int neighborIndex=csr.csrV[v];neighborIndex<csr.csrV[v+1];neighborIndex++){ //看該點鄰居
                    int neighborNodeID = csr.csrE[neighborIndex];

                    for(int multi_node=0;multi_node<mappingCount;multi_node++){
                        if(!s[layer][size].traverse_S[multi_node]) continue;

                        int position_n = mappingCount*neighborNodeID+multi_node;
                        int position   = mappingCount*v+multi_node;

                        if( dist_MULTI[position_n] == (dist_MULTI[position] + 1) ){
                            delta_MULTI[position] += (sigma_MULTI[position]/sigma_MULTI[position_n])*(1+delta_MULTI[position_n]);
                        }
                    }
                

                }
                // printf("delta_MULTI:  ");
                // for(auto i=0;i<multi_size;i++){
                //     printf("%f ",delta_MULTI[i]);
                //     if(i%2)
                //         printf("  ");
                // }
                // printf("\n");

            }
        }
        multi_time2 = seconds();
        multi_backward_Time +=(multi_time2-multi_time1);
        //累加delta至BC
        for (int sourceID = csr.startNodeID; sourceID <= csr.endNodeID; ++sourceID) {
            for(int multi_node=0; multi_node<mappingCount ; multi_node++ ){
                int position = mappingCount*sourceID+multi_node;
                BC[sourceID]+=delta_MULTI[position];
            }
        }


        #ifdef DEBUGx
            printf("dist_MULTI:  ");
            for(auto i=0;i<multi_size;i++){
                printf("%d ",dist_MULTI[i]);
                if(i%2)
                    printf("  ");
            }
            printf("\n");

            printf("sigma_MULTI: ");
            for(auto i=0;i<multi_size;i++){
                printf("%d ",sigma_MULTI[i]);
                if(i%2)
                    printf("  ");
            }
            printf("\n");
            
        #endif

    }
    multi_time_end = seconds();
    multi_total_time = (multi_time_end - multi_time_start);
    // Free memory
    free(s_size);
    free(dist_MULTI);
    free(sigma_MULTI);
    free(delta_MULTI);
    free(visited_MULTI);
}

void check_ans(vector<float> ans, vector<float> my_ans) {
    bool all_correct = true;
    float epsilon = 0;  // 定義一個很小的閾值

    for (size_t i = 0; i < ans.size(); i++) {
        // 將差異計算限制到小數點第二位
         // 將差異計算限制到小數點第二位
        float delta = ans[i] - my_ans[i];
        // // 取絕對值
        // delta = (delta < 0) ? -delta : delta;
        // // 模擬四捨五入到小數點後兩位
        // delta = (int)(delta * 100 + 0.5) / 100.0;

        
        
        
        if (delta > epsilon) {
            std::cout << "[ERROR] ans[" << i << "]= " << ans[i] << ", my_ans[" << i << "]= " << my_ans[i] << std::endl;
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "[CORRECT]  my_ans=ans!!\n";
    }

    return;
}


int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);

    vector<float> ans(csr->csrVSize,0);
    vector<float> my_BC(csr->csrVSize,0);
    //brandes start
    printf("csrVSize: %d\n",csr->csrVSize);
    printf("startNodeID: %d\n",csr->startNodeID);
    printf("endNodeID: %d\n",csr->endNodeID);
    printf("startAtZero: %d\n",csr->startAtZero);

    brandes_ORIGIN(*csr,csr->csrVSize,ans);
    // computeBC_shareBased(csr,my_BC);
    int max_multi=64;
    Seq_multi_source_brandes( *csr , max_multi , my_BC );
    check_ans(ans,my_BC);
    
    // for(auto i=0;i<csr->csrVSize;i++){
    //         printf("my_BC[%d]: %f\n",i,my_BC[i]);
    //     }
    #ifdef DEBUG
        for(auto i=0;i<csr->csrVSize;i++){
            printf("BC[%d]: %f\n",i,ans[i]);
        }
    #endif
    //brandes end
    // showCSR(csr);
    printf("\n=================================single_source run time=================================\n");
    printf("[Execution Time] forward_Time  = %.6f, %.6f \n", forward_Time, forward_Time / total_time);
    printf("[Execution Time] backward_Time = %.6f, %.6f \n", backward_Time, backward_Time / total_time);
    printf("[Execution Time] total_time    = %.6f, %.6f \n", total_time, total_time / total_time);
    printf("\n=================================multi_source run time=================================\n");
    printf("[Execution Time] forward_Time  = %.6f, %.6f \n", multi_forward_Time, multi_forward_Time / multi_total_time);
    printf("[Execution Time] backward_Time = %.6f, %.6f \n", multi_backward_Time, multi_backward_Time / multi_total_time);
    printf("[Execution Time] total_time    = %.6f, %.6f \n", multi_total_time, multi_total_time / multi_total_time);
    return 0;
}
