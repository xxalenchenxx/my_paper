#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <iomanip>  // 需要包含這個頭文件以使用 setprecision
#include <cuda_runtime.h>
#include "device_atomic_functions.h"
#endif

#include <vector>
#include <queue>
using namespace std;
#include "headers.h"
#define INFINITE 1000000000
// #define DEBUGx
// #define DEBUG

// Updated Q_struct definition
typedef struct q_struct {
    uint64_t traverse_S;
    int nodeID;
} Q_struct;

#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)


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

double time_start                       = 0;
double time_end                         = 0;
double time1                            = 0;
double time2                            = 0;

double multi_time_start                 = 0;
double multi_time_end                   = 0;
double multi_time1                      = 0;
double multi_time2                      = 0;

double mymethod_start                 = 0;
double mymethod_end                   = 0;
double mymethod_time1                      = 0;
double mymethod_time2                      = 0;


#pragma endregion //globalVar


inline void resetQueue(struct qQueue* _Q){
    _Q->front   = 0;
    _Q->rear    = -1;
    //Q->size如果不變，就不須memcpy
}

//原版brandes
void check_ans(std::vector<float> ans, std::vector<float> my_ans);
void brandes_ORIGIN_for_Seq( CSR& csr, int V, vector<float> &BC);
void brandes_with_predecessors(CSR& csr, int V, float* BC);
void brandes_with_predecessors_dynamic_check_ans(CSR csr, int V,int sourceID_test, vector<float> BC_ckeck);

void computeCC_ans(struct CSR* _csr, int* _CCs);
void compute_diameter(CSR* _csr);
void Seq_multi_source_brandes_ordered( CSR& csr, int max_multi, vector<float> &BC);
void Seq_multi_source_brandes(CSR& csr, int max_multi, vector<float> &BC);

//single source以及multi-source平行版本
void brandes_SS_par( CSR& csr, int V, float *BC);
void brandes_MS_par( CSR& csr, int max_multi, float* BC);
void brandes_MS_par_VnextQ( CSR& csr, int max_multi, float* BC);
void Seq_MS_brandes_me_D1_AP(CSR* csr, int max_multi, vector<float> &BC);
void brandes_MS_Me_AP_D1( CSR& csr, int max_multi, float* BC);

//test 程式
void computeCC_shareBased_oneTraverse(struct CSR* _csr, int* _CCs);
void computeBC_shareBased_Successor_SS( CSR* _csr, float* _BCs);
void computeBC_shareBased_Successor_SS_edge_update( CSR* _csr, float* _BCs);
void computeBC_shareBased_Successor_SS_test( CSR* _csr, float* _BCs);
void computeBC_shareBased_Successor_MS( CSR* _csr, float* _BCs);

void printbinary(int data,int mappingcount){
    int count = mappingcount;
    while(count--){
        int byte = (data>>count) &1;
        printf("%d",byte);
        if(count%8==0){
            printf(" ");
        }
    }
    printf("]\n");
}


int main(int argc, char* argv[]){
    if (argc < 3) {
        cout << "Error: insufficient input arguments.\n";
        cout << "Usage: " << argv[0] << " <datasetPath> <max_multi>\n";
        return 1;
    }
    char* datasetPath = argv[1];
    int max_multi=stoi(argv[2]);
    printf("exeName = %s\n", argv[0]);
    printf("datasetPath = %s\n", datasetPath);
    struct Graph* graph = buildGraph(datasetPath);
    struct CSR* csr     = createCSR(graph);

    vector<float> ans(csr->csrVSize,0.0);
    int *ans_CC= (int*)calloc(csr->csrVSize, sizeof(int));
    int *my_CC= (int*)calloc(csr->csrVSize, sizeof(int));
    float *ans_para= (float*)calloc(csr->csrVSize, sizeof(float));
    float *ans_para2= (float*)calloc(csr->csrVSize, sizeof(float));
    vector<float> my_BC(csr->csrVSize,0.0);
    vector<float> ans_para_vec(csr->csrVSize,0.0);
    vector<float> ans_para_vec2(csr->csrVSize,0.0);
    //brandes start
    printf("csrVSize: %d\n",csr->csrVSize);
    printf("startNodeID: %d\n",csr->startNodeID);
    printf("endNodeID: %d\n",csr->endNodeID);
    printf("startAtZero: %d\n",csr->startAtZero);
    // int max_multi=32;
    // compute_diameter(csr);

    time1 = seconds();
    // computeCC_shareBased_oneTraverse(csr,my_CC);
    // cout<<"max_degree: "<<csr->maxDegree<<endl;

    // brandes_ORIGIN_for_Seq(*csr,csr->csrVSize,ans);

    // brandes_SS_par(*csr,csr->csrVSize,ans_para);
    // brandes_MS_par(*csr , max_multi , ans_para);
    // // brandes_MS_par(*csr , max_multi , ans_para);
    // brandes_with_predecessors(*csr,csr->csrVSize,ans_para);
    // time2 = seconds();
    printf("done 1\n");


    multi_time1 = seconds();
    // computeCC_ans(csr,ans_CC);

    brandes_with_predecessors(*csr,csr->csrVSize,ans_para);
    

    // computeBC_shareBased_Successor_SS_test(csr,ans_para2);
    // computeBC_shareBased_Successor_MS(csr,ans_para2);
    // Seq_multi_source_brandes_ordered( *csr , max_multi , my_BC );
    // brandes_ORIGIN_for_Seq(*csr,csr->csrVSize,my_BC);
    // brandes_SS_par(*csr,csr->csrVSize,ans_para);
    // brandes_MS_par_VnextQ(*csr , max_multi , ans_para2); 
    // brandes_MS_Me_para(*csr , max_multi , ans_para2); 
    // Seq_MS_brandes_me_D1_AP(csr , max_multi , ans_para_vec); 
    // multi_time2 = seconds();
    printf("done 2\n");

    // computeBC_shareBased(csr,my_BC);
    // Seq_multi_source_brandes( *csr , max_multi , my_BC );

    mymethod_time1 = seconds();
    computeBC_shareBased_Successor_SS(csr,ans_para2);
    // computeBC_shareBased_Successor_SS_edge_update(csr,ans_para2);
    // mymethod_time2 = seconds();
    printf("done 3\n");

    //檢查答案BC
    // for(int i=0;i<csr->csrVSize;i++){
    //     ans_para_vec[i]=ans_para[i];
    //     ans_para_vec2[i]=ans_para2[i];
    // }
    // // check_ans(ans_para_vec,ans_para_vec2);
    // check_ans(ans_para_vec,ans_para_vec2);
    
    //答案檢查CC
    // bool flag=true;
    // for(int node = csr->startNodeID;node<=csr->endNodeID;node++){
    //     if(ans_CC[node]!=my_CC[node]){
    //         printf("[ERROR] ans[%d]:%d\tmy[%d]:%d\n",node,ans_CC[node],node,my_CC[node]);
    //         flag=false;
    //     }
    // }
    // if(flag){
    //     cout<<"[CORRECT] CC!!!\n";
    // }


    // #ifdef DEBUG
    //     for(auto i=0;i<csr->csrVSize;i++){
    //         printf("BC[%d]: %f\n",i,ans[i]);
    //     }
    // #endif
    //brandes end
    // showCSR(csr);
    // printf("\n=================================single_source run time=================================\n");
    // printf("[Execution Time] forward_Time  = %.6f, %.6f \n", forward_Time, forward_Time / total_time);
    // printf("[Execution Time] backward_Time = %.6f, %.6f \n", backward_Time, backward_Time / total_time);
    // printf("[Execution Time] total_time    = %.6f, %.6f \n", total_time, (forward_Time+backward_Time )/ total_time);
    // printf("\n=================================multi_source run time=================================\n");
    // printf("[Execution Time] forward_Time  = %.6f, %.6f \n", multi_forward_Time, multi_forward_Time / multi_total_time);
    // printf("[Execution Time] backward_Time = %.6f, %.6f \n", multi_backward_Time, multi_backward_Time / multi_total_time);
    // printf("[Execution Time] total_time    = %.6f, %.6f \n", multi_total_time, (multi_forward_Time + multi_backward_Time ) / multi_total_time);
    
    
    printf("[Execution Time] No_pred_total_time        = %.6f secs\n", time2-time1);
    printf("[Execution Time] pred_total_time           = %.6f secs\n", multi_time2-multi_time1);
    printf("[Execution Time] my_total_time             = %.6f secs\n", mymethod_time2-mymethod_time1);
    printf("[Execution Time] speedup(NO pred) ratio           = %.6f secs\n", (time2-time1)/(mymethod_time2-mymethod_time1));
    printf("[Execution Time] speedup2(Pred) ratio             = %.6f secs\n", (multi_time2-multi_time1)/(mymethod_time2-mymethod_time1));
    return 0;
}

void check_ans( std::vector<float> ans, std::vector<float> my_ans) {
    if (ans.size() != my_ans.size()) {
        std::cout << "[ERROR] Vectors have different sizes: ans.size()=" << ans.size()
                  << ", my_ans.size()=" << my_ans.size() << std::endl;
        return;
    }

    bool all_correct = true;
    float epsilon = 0.01;  // 定義誤差率為 1%

    for (size_t i = 0; i < ans.size(); i++) {
        // 計算絕對誤差
        float delta = std::fabs(ans[i] - my_ans[i]);
        // 計算允許的誤差範圍
        float error_rate = std::fabs(ans[i]) * epsilon;  // 基於 ans[i] 的相對誤差
        // float error_rate =0.0;  // 基於 ans[i] 的相對誤差

        if (delta > error_rate) {
            // 顯示完整的小數精度
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "[ERROR] ans[" << i << "] = " << ans[i]
                      << ", my_ans[" << i << "] = " << my_ans[i]
                      << ", delta = " << delta << ", allowed error = " << error_rate
                      << std::endl;
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "[CORRECT] my_ans matches ans " << std::endl;
    }

    return;
}

void quicksort_nodeID_with_degree(int* _nodes, int* _nodeDegrees, int _left, int _right){
    if(_left > _right){
        return;
    }
    int smallerAgent = _left;
    int smallerAgentNode = -1;
    int equalAgent = _left;
    int equalAgentNode = -1;
    int largerAgent = _right;
    int largerAgentNode = -1;

    int pivotNode = _nodes[_right];
    // printf("pivot : degree[%d] = %d .... \n", pivotNode, _nodeDegrees[pivotNode]);
    int tempNode = 0;
    while(equalAgent <= largerAgent){
        #ifdef DEBUG
        // printf("\tsmallerAgent = %d, equalAgent = %d, largerAgent = %d\n", smallerAgent, equalAgent, largerAgent);
        #endif

        smallerAgentNode = _nodes[smallerAgent];
        equalAgentNode = _nodes[equalAgent];
        largerAgentNode = _nodes[largerAgent];
        
        #ifdef DEBUG
        // printf("\tDegree_s[%d] = %d, Degree_e[%d] = %d, Degree_l[%d] = %d\n", smallerAgentNode, _nodeDegrees[smallerAgentNode], equalAgentNode, _nodeDegrees[equalAgentNode], largerAgentNode, _nodeDegrees[largerAgentNode]);
        #endif

        if(_nodeDegrees[equalAgentNode] > _nodeDegrees[pivotNode]){ //equalAgentNode的degree < pivotNode的degree
            // swap smallerAgentNode and equalAgentNode
            tempNode = _nodes[smallerAgent];
            _nodes[smallerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            smallerAgent ++;
            equalAgent ++;
        }
        else if(_nodeDegrees[equalAgentNode] < _nodeDegrees[pivotNode]){ //equalAgentNode的degree > pivotNode的degree
            // swap largerAgentNode and equalAgentNode
            tempNode = _nodes[largerAgent];
            _nodes[largerAgent] = _nodes[equalAgent];
            _nodes[equalAgent] = tempNode;

            largerAgent --;
        }
        else{ //equalAgentNode的degree == pivotNode的degree
            equalAgent ++;
        }

    }
    
    // exit(1);
    #ifdef DEBUG
        
    #endif

    // smallerAgent現在是pivot key的開頭
    // largerAgent現在是pivotKey的結尾
    quicksort_nodeID_with_degree(_nodes, _nodeDegrees, _left, smallerAgent - 1);
    quicksort_nodeID_with_degree(_nodes, _nodeDegrees, largerAgent + 1, _right);
}



//************************************************ */
//                   循序_brandes SS原版
//************************************************ */
 #pragma region brandes //forward traverse
// void brandes_ORIGIN_for_Seq(CSR& csr, int V, std::vector<float>& BC) {
//     // time1 = seconds();

//     double time_phase1=0.0;
//     double time_phase2=0.0;
//     double start_time=0.0;
//     double end_time=0.0;



//     // Allocate memory for sigma, dist, delta, and the stack S using STL containers
//     std::vector<std::vector<int>> S(V);       // S is a 2D stack
//     std::vector<int> sigma(V, 0);            // Sigma array
//     std::vector<int> dist(V, -1);            // Distance array
//     std::vector<float> delta(V, 0.0);        // Delta array
//     std::vector<int> S_size(V, 0);           // Stack size for each level
//     std::vector<int> f1(V);                  // Current frontier
//     std::vector<int> f2(V);                  // Next frontier

//     for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
//         start_time=seconds();
//         // Initialize variables for each source node
//         std::fill(sigma.begin(), sigma.end(), 0);
//         std::fill(dist.begin(), dist.end(), -1);
//         std::fill(delta.begin(), delta.end(), 0.0);
//         std::fill(S_size.begin(), S_size.end(), 0);
//         for (auto& level : S) {
//             level.clear();
//         }

//         sigma[s] = 1;
//         dist[s] = 0;
//         int f1_indicator = 0;
//         int f2_indicator = 0;

//         // Re-initialize current_queue
//         f1[f1_indicator++] = s;

//         int level = 0;
//         // BFS forward phase: frontier-based BFS
//         while (f1_indicator > 0) {
//             int* currentQueue = (level % 2 == 0) ? f1.data() : f2.data();
//             int* nextQueue = (level % 2 == 0) ? f2.data() : f1.data();

//             for (int v = 0; v < f1_indicator; ++v) {
//                 int u = currentQueue[v];
//                 S[level].push_back(u);  // Put node u into its level

//                 // Traverse the adjacent nodes in CSR format
//                 for (int i = csr.csrV[u]; i < csr.csrV[u + 1]; ++i) {
//                     int w = csr.csrE[i];

//                     // If w has not been visited, update distance and add to next_queue
//                     if (dist[w] < 0) {
//                         dist[w] = dist[u] + 1;
//                         nextQueue[f2_indicator++] = w;
//                     }

//                     // When a shortest path is found
//                     if (dist[w] == dist[u] + 1) {
//                         sigma[w] += sigma[u];
//                     }
//                 }
//             }

//             // Prepare for the next iteration
//             f1_indicator = f2_indicator;
//             f2_indicator = 0;
//             level++;
//         }

//         end_time=seconds();
//         time_phase1+= end_time - start_time;
//         start_time=seconds();
//         // Backward phase to compute BC values
//         for (int d = level - 1; d >= 0; --d) {  // Start from the furthest level
//             for (int w : S[d]) {
//                 for (int i = csr.csrV[w]; i < csr.csrV[w + 1]; ++i) {
//                     int v = csr.csrE[i];
//                     if (dist[v] == dist[w] - 1) {
//                         delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
//                     }
//                 }
//                 if (w != s) {
//                     BC[w] += delta[w];
//                 }
//             }
//         }
//         end_time=seconds();
//         time_phase2+= end_time - start_time;
//     }
//     printf("phase1 time: %0.6f\n", time_phase1);
//     printf("phase2 time: %0.6f\n", time_phase2);
//     // time2 = seconds();
// }


void brandes_ORIGIN_for_Seq(CSR& csr, int V, std::vector<float>& BC) {
    // Time measurement
    double time_phase1 = 0.0;
    double time_phase2 = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;

    // Allocate memory for sigma, dist, delta, and the stack S using STL containers
    std::vector<std::vector<int>> S(V);       // S is a 2D stack
    std::vector<int> sigma(V, 0);            // Sigma array
    std::vector<int> dist(V, -1);            // Distance array
    std::vector<float> delta(V, 0.0);        // Delta array
    std::queue<int> f1;                      // Current frontier queue
    std::queue<int> f2;                      // Next frontier queue

    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        start_time = seconds();
        
        // Initialize variables for each source node
        std::fill(sigma.begin(), sigma.end(), 0);
        std::fill(dist.begin(), dist.end(), -1);
        std::fill(delta.begin(), delta.end(), 0.0);
        S.assign(V, vector<int>());  // Reset Successors with empty vectors
        sigma[s] = 1;
        dist[s] = 0;

        // Initialize the frontier
        f1.push(s);

        int level = 0;

        // BFS forward phase
        while (!f1.empty()) {
            while (!f1.empty()) {
                int u = f1.front();
                f1.pop();
                S[level].push_back(u);  // Put node u into its level

                // Traverse the adjacent nodes in CSR format
                for (int i = csr.csrV[u]; i < csr.csrV[u + 1]; ++i) {
                    int w = csr.csrE[i];

                    // If w has not been visited, update distance and add to next frontier
                    if (dist[w] < 0) {
                        dist[w] = dist[u] + 1;
                        f2.push(w);
                    }

                    // When a shortest path is found
                    if (dist[w] == dist[u] + 1) {
                        sigma[w] += sigma[u];
                    }
                }
            }

            // Swap queues for the next level
            std::swap(f1, f2);
            level++;
        }

        end_time = seconds();
        time_phase1 += end_time - start_time;
        start_time = seconds();

        // Backward phase to compute BC values
        for (int d = level - 1; d >= 0; --d) {  // Start from the furthest level
            for (int w : S[d]) {
                for (int i = csr.csrV[w]; i < csr.csrV[w + 1]; ++i) {
                    int v = csr.csrE[i];
                    if (dist[v] == dist[w] - 1) {
                        delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
                    }
                }
                if (w != s) {
                    BC[w] += delta[w];
                }
            }
        }

        end_time = seconds();
        time_phase2 += end_time - start_time;
    }
    time2 += time1 + time_phase1 + time_phase2;
    printf("phase1 time: %0.6f\n", time_phase1);
    printf("phase2 time: %0.6f\n", time_phase2);
}


#pragma endregion




//循序_brandes切D1與AP
void Seq_MS_brandes_me_D1_AP(CSR* csr, int max_multi, vector<float> &BC) {
    // Start timing
    // multi_time_start = seconds();

    int v_size = csr->csrVSize;
    int* map_S = (int*)malloc(sizeof(int) * max_multi); // Multiple sources
    bool* nodeDone = (bool*)calloc(v_size, sizeof(bool));

    size_t multi_size = v_size * max_multi;

    int* s_size = (int*)malloc(sizeof(int) * v_size);
    float* dist_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* sigma_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* delta_MULTI = (float*)malloc(sizeof(float) * multi_size);

    Q_struct** s = (Q_struct**)malloc(v_size * sizeof(Q_struct*));
    Q_struct* f1 = (Q_struct*)malloc(v_size * sizeof(Q_struct));
    Q_struct* f2 = (Q_struct*)malloc(v_size * sizeof(Q_struct));

    // Pre-initialize dist_MULTI to INFINITE
    float* dist_INIT = (float*)malloc(sizeof(float) * multi_size);
    
    for (size_t i = 0; i < multi_size; i++) {
        dist_INIT[i] = INFINITE;
    }
    // showCSR(csr);
    //D1 Folding
    D1Folding(csr);
    // for(int i=0;i<csr->csrVSize;i++){
    //     std::cout<<"_csr->representNode["<<i<<"]: "<<csr->representNode[i]<<"\t_csr->ff["<<i<<"]: "<<csr->ff[i]<<endl;
    // }

    // showCSR(csr);
    //標記那些node是AP
    AP_detection(csr);
    //做3case的分割
    AP_Copy_And_Split_BC(csr);

    // struct newID_info* newID_infos = rebuildGraph(csr);
    // for(int i=0;i< csr->ap_count;i++){
    //     cout<<csr->AP_List[i]<<" ";
    // }
    // cout<<"csr.ap_count: "<<csr->ap_count<<"\n";
    // multi_time_end = seconds();
    // multi_total_time = (multi_time_end - multi_time_start);

    // Free memory
    free(s_size);
    free(dist_MULTI);
    free(sigma_MULTI);
    free(delta_MULTI);
    free(map_S);
    free(nodeDone);
    free(s);
    free(f1);
    free(f2);
    free(dist_INIT);
}


//************************************************ */
//                   循序_brandes 測試
//************************************************ */

void brandes_with_predecessors(CSR& csr, int V, float* BC) {
    double time_phase1=0.0;
    double time_phase2=0.0;
    double start_time=0.0;
    double end_time=0.0;
    
    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);               // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> predecessors(V);   // Predecessor list

    long long total_predecessor_count = 0; // To accumulate total predecessors

    

    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        start_time=seconds();
        // Initialize arrays for each source node
        // sigma.assign(V, 0);   // Reset sigma to size V with all values 0
        // dist.assign(V, -1);   // Reset dist to size V with all values -1
        // delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
        fill(sigma.begin(), sigma.end(), 0);
        fill(dist.begin(), dist.end(), -1);
        fill(delta.begin(), delta.end(), 0.0);
        S.assign(V, vector<int>());  // Reset S with empty vectors
        predecessors.assign(V, vector<int>());  // Reset Successors with empty vectors
        // for (auto& level : S) {
        //     level.clear();
        // }
        // for (auto& preds : predecessors) {
        //     preds.clear();
        // }
        

        sigma[s] = 1;
        dist[s] = 0;
        f1.push(s);

        int level = 0;

        // BFS forward phase
        while (!f1.empty()) {
            while (!f1.empty()) {
                int u = f1.front();
                f1.pop();
                S[level].push_back(u);

                // Traverse neighbors in CSR
                for (int i = csr.csrV[u]; i < csr.csrV[u + 1]; ++i) {
                    int w = csr.csrE[i];

                    if (dist[w] < 0) {
                        dist[w] = dist[u] + 1;
                        f2.push(w);
                    }

                    if (dist[w] == dist[u] + 1) {
                        sigma[w] += sigma[u];
                        predecessors[w].push_back(u);
                    }
                }
            }
            swap(f1, f2);
            level++;
        }
        end_time=seconds();
        time_phase1 += end_time - start_time;
        start_time=seconds();
        // Backward phase
        for (int d = level - 1; d >= 0; --d) {
            for (int w : S[d]) {
                for (int v : predecessors[w]) {
                    delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
                }
                if (w != s) {
                    BC[w] += delta[w];
                }
            }
        }

        end_time=seconds();
        time_phase2 += end_time - start_time;
        // Accumulate total predecessors
        // for (const auto& preds : predecessors) {
        //     total_predecessor_count += preds.size();
        // }
    }
    multi_time2 += multi_time1+time_phase1+time_phase2;
    printf("phase1 time: %0.6f\n", time_phase1);
    printf("phase2 time: %0.6f\n", time_phase2);
    // for(int i=csr.startNodeID;i<= csr.endNodeID; i++){
    //     printf("BC_ans[%d]: %0.2f\n",i,BC[i]);
    // }
    // Calculate and print average predecessors
    // double average_predecessors = (double)total_predecessor_count / V; //每個點當source時平均的pred edge數量。
    // cout << "avg  pred: " << average_predecessors << endl;
    // cout << "edge size: " << csr.csrESize << endl;
    // cout << "\n---------------------\n" << endl;
    // cout << "percentag edge: " << (float)average_predecessors / csr.csrESize *100<< endl; //pred edge/比例。
    // cout << "percentag node: " << (float)csr.csrVSize / csr.csrESize *100<< endl;
    // cout << "avg pred edge of node: " << (float)csr.csrESize / average_predecessors << endl;
    // cout << "avg edge of node: " << (float)csr.csrESize / csr.csrVSize << endl;
    // cout << "\n---------------------\n" << endl;
}

void brandes_with_predecessors_dynamic_check_ans(CSR csr, int V,int sourceID_test, vector<float> BC_ckeck) {
    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);               // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> predecessors(V);   // Predecessor list

    int s = sourceID_test;
    vector<float> BC(V, 0);
        // Initialize arrays for each source node
        fill(sigma.begin(), sigma.end(), 0);
        fill(dist.begin(), dist.end(), -1);
        fill(delta.begin(), delta.end(), 0.0);
        for (auto& level : S) {
            level.clear();
        }
        for (auto& preds : predecessors) {
            preds.clear();
        }

        sigma[s] = 1;
        dist[s] = 0;
        f1.push(s);

        int level = 0;

        // BFS forward phase
        while (!f1.empty()) {
            while (!f1.empty()) {
                int u = f1.front();
                f1.pop();
                S[level].push_back(u);

                // Traverse neighbors in CSR
                for (int i = csr.csrV[u]; i < csr.csrV[u + 1]; ++i) {
                    int w = csr.csrE[i];

                    if (dist[w] < 0) {
                        dist[w] = dist[u] + 1;
                        f2.push(w);
                    }

                    if (dist[w] == dist[u] + 1) {
                        sigma[w] += sigma[u];
                        predecessors[w].push_back(u);
                    }
                }
            }
            swap(f1, f2);
            level++;
        }

        // Backward phase
        for (int d = level - 1; d >= 0; --d) {
            for (int w : S[d]) {
                for (int v : predecessors[w]) {
                    delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
                }
                if (w != s) {
                    BC[w] += delta[w];
                }
            }
        }

    bool flag=true;
    for(int i=0;i<delta.size();i++){
        if(delta[i]!=BC_ckeck[i]){
            printf("[ERROR] ans[%d]: %0.2f my_ans[%d]: %0.2f \n",i,delta[i],i,BC_ckeck[i]);
            flag=false;
        }
    }  
    if(flag)
        printf("[COORRECT] dynamic ans\n");

}


void computeCC_ans(struct CSR* _csr, int* _CCs){
    // showCSR(_csr);
    int* dist_arr       = (int*)calloc(sizeof(int), _csr->csrVSize);

    struct qQueue* Q    = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    int sourceID;

    #ifdef CheckDistAns
    sourceID = tempSourceID;
    int CC_ans = 0;
    #else
    sourceID = _csr->startNodeID;
    // sourceID = 1;
    #endif

    for(; sourceID <= _csr->endNodeID ; sourceID ++){
        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        resetQueue(Q);

        qPushBack(Q, sourceID);
        dist_arr[sourceID]  = 0;

        // #ifdef DEBUG
        // printf("\nSourceID = %2d ...\n", sourceID);
        // #endif      

        int currentNodeID   = -1;
        int neighborNodeID  = -1;


        while(!qIsEmpty(Q)){
            currentNodeID = qPopFront(Q);
            
            // #ifdef DEBUG
            // printf("%2d ===\n", currentNodeID);
            // #endif

            for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->oriCsrV[currentNodeID + 1] ; neighborIndex ++){
                neighborNodeID = _csr->csrE[neighborIndex];

                // #ifdef DEBUG
                // printf("\t%2d meet %2d, dist_arr[%2d] = %2d\n", currentNodeID, neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                // #endif

                if(dist_arr[neighborNodeID] == -1){
                    qPushBack(Q, neighborNodeID);
                    dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

                    // #ifdef DEBUG
                    // printf("\tpush %2d to Q, dist_arr[%2d] = %2d\n", neighborNodeID, neighborNodeID, dist_arr[neighborNodeID]);
                    // #endif
                    
                    #ifdef CheckDistAns
                    CC_ans += dist_arr[neighborNodeID];
                    #else
                    _CCs[sourceID] += dist_arr[neighborNodeID];
                    #endif

                }
            }
        }
        
        // break;
        #ifdef CheckDistAns
        printf("CC[%d] = %d\n", tempSourceID, CC_ans);
        break;
        #endif
        
    }

    free(Q->dataArr);
    free(Q);

    #ifndef CheckDistAns
    free(dist_arr);
    #endif

    return;
}

void compute_diameter(CSR* _csr) {
    int V = _csr->csrVSize;
    vector<int> dist(V, -1); // Distance array for BFS
    int diameter = 0;        // To store the graph diameter
    pair<int, int> farthestNodes; // To store the two nodes that are farthest apart

    for (int sourceID = _csr->startNodeID; sourceID <= _csr->endNodeID; sourceID++) {
        fill(dist.begin(), dist.end(), -1); // Reset distances
        queue<int> q;

        // BFS initialization
        q.push(sourceID);
        dist[sourceID] = 0;

        int maxDist = 0;
        int farthestNode = sourceID;

        // BFS to calculate shortest paths
        while (!q.empty()) {
            int currentNode = q.front();
            q.pop();

            for (int neighborIndex = _csr->csrV[currentNode]; neighborIndex < _csr->csrV[currentNode + 1]; neighborIndex++) {
                int neighborNode = _csr->csrE[neighborIndex];
                if (dist[neighborNode] == -1) { // Unvisited node
                    dist[neighborNode] = dist[currentNode] + 1;
                    q.push(neighborNode);

                    if (dist[neighborNode] > maxDist) {
                        maxDist = dist[neighborNode];
                        farthestNode = neighborNode;
                    }
                }
            }
        }

        // Update diameter and record farthest nodes
        if (maxDist > diameter) {
            diameter = maxDist;
            farthestNodes = {sourceID, farthestNode};
        }
    }

    // Output the result
    cout << "Diameter of the graph: " << diameter << endl;
    // cout << "Farthest nodes: (" << farthestNodes.first << ", " << farthestNodes.second << ")" << endl;
}

//用來分析Vw U1 U2的分布
void computeCC_shareBased_oneTraverse(struct CSR* _csr, int* _CCs){
    // showCSR(_csr);
    
    int* dist_arr           = (int*)malloc(sizeof(int) * _csr->csrVSize);
    int* neighbor_dist_ans  = (int*)malloc(sizeof(int) * _csr->csrVSize);

    struct qQueue* Q        = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //record that nodes which haven't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);
    
    long long Vw_count=0,U1_count=0,U2_count=0; //計算VW數量
    long long margin_count=0;
    long long shared_source_count=0;
    long long update_pred_degree=0;
    //record nodes belongs to which neighbor of source
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    unsigned int* sharedBitIndex    = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize); //for recording blue edge bitIndex
    unsigned int* relation          = (unsigned int*)calloc(sizeof(unsigned int), _csr->csrVSize); //for recording red edge bitIndex
    
    //order ID by degree
    _csr->orderedCsrV  = (int*)calloc(sizeof(int), (_csr->csrVSize) *2);
    for(int i=_csr->startNodeID;i<=_csr->endNodeID;i++){
            _csr->orderedCsrV[i]=i;
    }
    quicksort_nodeID_with_degree(_csr->orderedCsrV, _csr->csrNodesDegree, _csr->startNodeID, _csr->endNodeID);

    for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
    // for(int sourceIDIndex = _csr->startNodeID ; sourceIDIndex <= _csr->startNodeID ; sourceIDIndex ++){
    //     int sourceID = _csr->orderedCsrV[sourceIDIndex];
        if(nodeDone[sourceID] == 1){
            continue;
        }
        nodeDone[sourceID] = 1;

        // printf("SourceID = %2d\n", sourceID);

        memset(dist_arr, -1, sizeof(int) * _csr->csrVSize);
        
        resetQueue(Q);
        
        dist_arr[sourceID] = 0;
        qPushBack(Q, sourceID);

        int currentNodeID  = -1;
        int neighborNodeID = -1;
        int neighborIndex  = -1;
        
        //each neighbor of sourceID mapping to bit_SI, if it haven't been source yet
        int mappingCount = 0;
        for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(nodeDone[neighborNodeID] == 0){
                shared_source_count++;
                sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                mapping_SI[mappingCount] = neighborNodeID;

                // printf("sharedBitIndex[%6d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #ifdef DEBUG
                printf("sharedBitIndex[%2d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #endif
                
                mappingCount ++;

                //Record to 32 bit only
                if(mappingCount == 1){
                    break;
                }

            }
        }
        
        // if(mappingCount < 3){
        //     //把sharedBitIndex重設。
        //     for(int mappingIndex = 0 ; mappingIndex < mappingCount ; mappingIndex ++){
        //         int nodeID = mapping_SI[mappingIndex];
        //         sharedBitIndex[nodeID] = 0;
        //     }
        //     memset(mapping_SI, 0, sizeof(int) * 32);

        //     #pragma region Ordinary_BFS_Forward_Traverse

        //     #ifdef DEBUG
        //     printf("\n####      Source %2d Ordinary BFS Traverse      ####\n\n", sourceID);
        //     #endif

        //     while(!qIsEmpty(Q)){
        //         currentNodeID = qPopFront(Q);

        //         #ifdef DEBUG
        //         printf("\tcurrentNodeID = %2d ... dist = %2d\n", currentNodeID, dist_arr[currentNodeID]);
        //         #endif

        //         for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
        //             neighborNodeID = _csr->csrE[neighborIndex];

        //             if(dist_arr[neighborNodeID] == -1){
        //                 qPushBack(Q, neighborNodeID);
        //                 dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;

        //                 #ifdef DEBUG
        //                 printf("\t\t[1]dist[%2d] = %2d\n", neighborNodeID, dist_arr[neighborNodeID]);
        //                 #endif
        //             }
        //         }
        //     }
        //     #pragma endregion //Ordinary_BFS_Forward_Traverse



        //     #pragma region distAccumulation_pushBased
        //     //Update CC in the way of pushing is better for parallelism because of the it will not need to wait atomic operation on single address,
        //     //it can update all value in each CC address in O(1) time.
        //     for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
        //         _CCs[nodeID] += dist_arr[nodeID];
        //     }
        //     #pragma endregion //distAccumulation_pushBased



        //     #pragma region checkingDistAns
        //     #ifdef CheckDistAns
        //     // CC_CheckDistAns(_csr, _CCs, sourceID, dist_arr);
        //     #endif

        //     #ifdef CheckCC_Ans
        //     dynamic_CC_trace_Ans(_csr, _CCs, sourceID);
        //     #endif

        //     #pragma endregion //checkingDistAns

        // }
        // else{

            #pragma region SourceTraverse
            //main source traversal : for getting the dist of each node from source
            // #ifdef DEBUG
            // printf("\n####      Source %2d First traverse...      ####\n\n", sourceID);
            // #endif

            while(!qIsEmpty(Q)){
                currentNodeID = qPopFront(Q);

                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d\n", currentNodeID, dist_arr[currentNodeID]);
                #endif

                

                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
                    neighborNodeID = _csr->csrE[neighborIndex];

                    if(dist_arr[neighborNodeID] == -1){//traverse new succesor and record its SI
                        qPushBack(Q, neighborNodeID);
                        dist_arr[neighborNodeID] = dist_arr[currentNodeID] + 1;
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];

                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif
                        
                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[1]currentNodeID = %2d(dist %2d, SI %2x), neighborNodeID = %d(dist %2d, SI %2x)\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], sharedBitIndex[neighborNodeID]);
                        // }
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){ //traverse to discovered succesor and record its SI
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];    
                        
                        #ifdef DEBUG
                        printf("\t[2]visited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist_arr[neighborNodeID]);
                        #endif

                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[2]currentNodeID = %2d(dist %2d, SI %2x), neighborNodeID = %d(dist %2d, SI %2x)\n", currentNodeID, dist_arr[currentNodeID], sharedBitIndex[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], sharedBitIndex[neighborNodeID]);
                        // }
                    }
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] && currentNodeID < neighborNodeID){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID]     |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                        relation[neighborNodeID]    |= sharedBitIndex[currentNodeID]  & (~sharedBitIndex[neighborNodeID]);

                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif

                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[3]currentNodeID = %2d(dist %2d, re %2x), neighborNodeID = %d(dist %2d, re %2x)\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], relation[neighborNodeID]);
                        // }
                    }//&& relation[neighborNodeID]
                    else if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] - 1){ //traverse to discovered neighbor which is at same level as currentNodeID
                        // relation[currentNodeID]     |= relation[neighborNodeID];
                        //這一步是讓SI 跟R只會顯示一個 SI:1 R:0 幫助最終判鄰居是否相同
                        relation[currentNodeID]     = (relation[currentNodeID]|relation[neighborNodeID]) & (~sharedBitIndex[currentNodeID] );
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif

                        // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
                        //     printf("\t[3]currentNodeID = %2d(dist %2d, re %2x), neighborNodeID = %d(dist %2d, re %2x)\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], relation[neighborNodeID]);
                        // }
                    }
                }
            }

            //second source traversal : for handle the red edge
            // #ifdef DEBUG
            // printf("\n####      Source %2d Second traverse...      ####\n\n", sourceID);
            // #endif

            // Q->front = 0;
            // while(!qIsEmpty(Q)){
            //     currentNodeID = qPopFront(Q);

            //     #ifdef DEBUG
            //     printf("currentNodeID = %2d ... dist = %2d ... relation = %x\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID]);
            //     #endif

            //     for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++){
            //         neighborNodeID = _csr->csrE[neighborIndex];

            //         if(dist_arr[neighborNodeID] == dist_arr[currentNodeID] + 1){
            //             relation[neighborNodeID] |= relation[currentNodeID];
                        
            //             #ifdef DEBUG
            //             printf("\t[4]relation[%2d] = %2x\n", neighborNodeID, relation[neighborNodeID]);
            //             #endif

            //             // if(sourceID == 5 && (neighborNodeID == 4 || neighborNodeID == 6)){
            //             //     printf("\t[4]currentNodeID = %2d(dist %2d, re %2x), neighborNodeID = %d(dist %2d, re %2x)\n", currentNodeID, dist_arr[currentNodeID], relation[currentNodeID], neighborNodeID, dist_arr[neighborNodeID], relation[neighborNodeID]);
            //             // }
            //         }
            //     }
            // }
            #pragma endregion //SourceTraverse

            #pragma region sourceDistAccumulation_pushBased
            for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                _CCs[nodeID] += dist_arr[nodeID];
            }

            #ifdef CheckCC_Ans
            dynamic_CC_trace_Ans(_csr, _CCs, sourceID);
            #endif

            #pragma endregion //distAccumulation_pushBased


            #pragma region neighborOfSource_GetDist
            //recover the data from source to neighbor of source
            for(int sourceNeighborIndex = 0 ; sourceNeighborIndex < mappingCount ; sourceNeighborIndex ++){
                memset(neighbor_dist_ans, 0, sizeof(int));

                int sourceNeighborID = mapping_SI[sourceNeighborIndex];
                unsigned int bit_SI = 1 << sourceNeighborIndex;

                nodeDone[sourceNeighborID] = 1;

                #ifdef DEBUG
                printf("\nnextBFS = %2d, bit_SI = %x\n", sourceNeighborID, bit_SI);
                #endif

                for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                    int nodeID_type=-1; //1: Vw -- 2:U1 -- 3:U2 
                    if((sharedBitIndex[nodeID] & bit_SI) > 0){ //要括號，因為"比大小優先於邏輯運算"
                        neighbor_dist_ans[nodeID] = dist_arr[nodeID] - 1;
                        Vw_count++;
                        nodeID_type=1;
                        // printf("\t[5]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, sharedBitIndex[nodeID]);
                    }
                    else{
                        neighbor_dist_ans[nodeID] = dist_arr[nodeID] + 1;
                        // printf("\t[6]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, sharedBitIndex[nodeID]);
                        if((relation[nodeID] & bit_SI) > 0){
                            U1_count++;
                            neighbor_dist_ans[nodeID] --;
                            nodeID_type=2;
                            // printf("\t[7]neighbor_dist_ans[%2d] = %2d, relation[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, relation[nodeID]);
                        }else{
                            nodeID_type=3;
                            U2_count++;
                        }
                    }

                    for(neighborIndex = _csr->csrV[nodeID] ; neighborIndex < _csr->csrV[nodeID + 1] ; neighborIndex ++){
                        neighborNodeID = _csr->csrE[neighborIndex];
                        int neighborID_type=-1;
                        if((sharedBitIndex[neighborNodeID] & bit_SI) > 0){ //要括號，因為"比大小優先於邏輯運算"
                            neighborID_type=1;
                            // printf("\t[5]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, sharedBitIndex[nodeID]);
                        }
                        else{
                            // printf("\t[6]neighbor_dist_ans[%2d] = %2d, SI[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, sharedBitIndex[nodeID]);
                            if((relation[nodeID] & bit_SI) > 0){
                                neighborID_type=2;
                                // printf("\t[7]neighbor_dist_ans[%2d] = %2d, relation[%2d] = %x\n", nodeID, neighbor_dist_ans[nodeID], nodeID, relation[nodeID]);
                            }else{
                                neighborID_type=3;
                            }
                        }
                        if(nodeID_type!=neighborID_type){
                            update_pred_degree += _csr->csrNodesDegree[nodeID];
                            margin_count++;
                            break;
                        }
                    }
                }

                #pragma region neighborDistAccumulation_pushBased
                for(int nodeID = _csr->startNodeID ; nodeID <= _csr->endNodeID ; nodeID ++){
                    _CCs[nodeID] += neighbor_dist_ans[nodeID];
                }
                #pragma endregion //neighborDistAccumulation_pushBased

            }


            
            #pragma endregion //neighborOfSource_GetDist

            //reset the SI & relation arrays
            memset(relation, 0, sizeof(unsigned int) * _csr->csrVSize);
            memset(sharedBitIndex, 0, sizeof(unsigned int) * _csr->csrVSize);
        // }
    }
    float avg_margin_node = margin_count/shared_source_count;
    float avg_pred_update_degree = update_pred_degree/shared_source_count;
    float avg_pred_update_degree_ratio = avg_pred_update_degree/_csr->csrESize;

    float avg_Vw_node = Vw_count/shared_source_count;
    float avg_U1_node = U1_count/shared_source_count;
    float avg_U2_node = U2_count/shared_source_count;
    // cout<<"avg_margin_node: "<<avg_margin_node<<" marg_node_ratio: "<<avg_margin_node/_csr->csrVSize*100<<endl;
    // cout<<"avg_marg_edge% : "<<avg_pred_update_degree<<" marg_edge_ratio: "<<avg_pred_update_degree_ratio*100<<endl;
    // cout<<"avg_Vw_node:     "<<avg_Vw_node<<" Vw_node_ratio: "<<avg_Vw_node/_csr->csrVSize*100<<endl;
    // cout<<"avg_U1_node:     "<<avg_U1_node<<" U1_node_ratio: "<<avg_U1_node/_csr->csrVSize*100<<endl;
    // cout<<"avg_U2_node:     "<<avg_U2_node<<" U2_node_ratio: "<<avg_U2_node/_csr->csrVSize*100<<endl;

    // printf("\n\n[CC_sharedBased] Done!\n");
}

//使用shared 和successor來BC (循序版本，平行模板)
void computeBC_shareBased_Successor_MS( CSR* _csr, float* _BCs){
    // showCSR(_csr);
    int V =  _csr->csrVSize;

    struct qQueue* Q        = InitqQueue();
    qInitResize(Q, _csr->csrVSize);

    //record that nodes which haven't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);
    
    //record nodes belongs to which neighbor of source
    int  mappingCount_max           = 32; //最大可以設64
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    vector<unsigned int> sharedBitIndex(_csr->csrVSize, 0); // for recording blue edge bitIndex
    vector<unsigned int> relation(_csr->csrVSize, 0);       // for recording red edge bitIndex
    vector<unsigned long long> sameIndex(_csr->csrVSize, 0);// for recording blue edge BC_sameIndex

    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);              // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);               // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<vector<int>> Successors(V);     // Successors list 

    //可重複使用的空間
    vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    queue<Q_struct>  Q_f1,Q_f2;
    vector<Q_struct> Q_f2_temp(V);
    //N(s)鄰居的traverse 路徑數量、距離、Suc、Sigma
    vector<vector<int>>   Ns_dist  (mappingCount_max, vector<int>(V,-1));               // Distance array
    vector<vector<int>>   Ns_sigma (mappingCount_max, vector<int>(V,0));
    vector<vector<float>> Ns_delta (mappingCount_max, vector<float>(V,0.0));            // Delta array
    vector<vector<vector<int>>> Ns_Successors(mappingCount_max,vector<vector<int>>(V)); // Successors [][][]: [source的Suc] [V] [V的Suc] 
    vector<vector<Q_struct>> Ns_S(V);               // S is a 2D stack [level][node]...


    //Suc與edge數據
    unsigned long Suc_times=0;
    unsigned long edge_times=0;
    //用degree做排序 大->小
    _csr->orderedCsrV  = (int*)calloc(sizeof(int), (_csr->csrVSize) *2);
    for(int i=_csr->startNodeID;i<=_csr->endNodeID;i++){
            _csr->orderedCsrV[i]=i;
    }
    quicksort_nodeID_with_degree(_csr->orderedCsrV, _csr->csrNodesDegree, _csr->startNodeID, _csr->endNodeID);

    // for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
    


    for(int sourceIDIndex = _csr->startNodeID ; sourceIDIndex <= _csr->endNodeID ; sourceIDIndex ++){
        int sourceID = _csr->orderedCsrV[sourceIDIndex];
        if(nodeDone[sourceID] == 1){
            continue;
        }
        nodeDone[sourceID] = 1;

        // printf("SourceID = %2d\n", sourceID);

        int currentNodeID  = -1;
        int neighborNodeID = -1;
        int neighborIndex  = -1;
        
        //each neighbor of sourceID mapping to bit_SI, if it haven't been source yet
        //挑選鄰居
        int mappingCount = 0;
        for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(nodeDone[neighborNodeID] == 0){
                //Record to 32 bit only
                if(mappingCount == mappingCount_max){
                    break;
                }

                sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                mapping_SI[mappingCount] = neighborNodeID;
                nodeDone[neighborNodeID] = 1;

                // printf("sharedBitIndex[%6d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #ifdef DEBUG
                printf("sharedBitIndex[%2d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #endif
                
                mappingCount ++;
            }
        }

        //**********************************/
        //   Source forward traverse
        //**********************************/
        #pragma region SourceTraverse //forward traverse
        
        // Initialize arrays for each source node
        sigma.assign(V, 0);   // Reset sigma to size V with all values 0
        dist.assign(V, -1);   // Reset dist to size V with all values -1
        delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
        S.assign(V, vector<int>());  // Reset S with empty vectors
        Successors.assign(V, vector<int>());  // Reset Successors with empty vectors

        sigma[sourceID] = 1;
        dist[sourceID] = 0;
        f1.push(sourceID);
        int level = 0;
        // cout<<"source ID: "<<sourceID<<endl;
        // BFS forward phase
        
        while (!f1.empty()){
            while (!f1.empty()) {
                int currentNodeID = f1.front();
                f1.pop();
                S[level].push_back(currentNodeID);
                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d\n", currentNodeID, dist[currentNodeID]);
                #endif
                // Traverse neighbors in CSR
                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++) {
                    int neighborNodeID =  _csr->csrE[neighborIndex];

                    if (dist[neighborNodeID] == -1) {
                        dist[neighborNodeID] = dist[currentNodeID] + 1;
                        sigma[neighborNodeID] += sigma[currentNodeID];
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        Successors[currentNodeID].push_back(neighborNodeID);
                        f2.push(neighborNodeID);
                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist[neighborNodeID]);
                        #endif
                    }
                    else if (dist[neighborNodeID] == dist[currentNodeID] + 1) {
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        Successors[currentNodeID].push_back(neighborNodeID);
                        sigma[neighborNodeID] += sigma[currentNodeID];
                        #ifdef DEBUG
                        printf("\t[2]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist[neighborNodeID]);
                        #endif
                    }
                    else if(dist[neighborNodeID] == dist[currentNodeID] && currentNodeID < neighborNodeID){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID]     |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                        relation[neighborNodeID]    |= sharedBitIndex[currentNodeID]  & (~sharedBitIndex[neighborNodeID]);
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif

                    }//&& relation[neighborNodeID]
                    else if(dist[neighborNodeID] == dist[currentNodeID] - 1){
                        //這一步是讓SI 跟R只會顯示一個 SI:1 R:0 幫助最終判鄰居是否相同 
                        relation[currentNodeID]     = (relation[currentNodeID]|relation[neighborNodeID]) & (~sharedBitIndex[currentNodeID] );
                        #ifdef DEBUG
                        printf("\t[4]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif
                    }
                }

            }
            swap(f1, f2);
            level++;
        }



        #ifdef DEBUG
            //printf successor
            printf("\n");
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("Successor(%d) sigma(%d) ",i,sigma[i] );
                for (int j=0;j<Successors[i].size();j++){
                    printf("%d ",Successors[i][j]);
                }
                printf("\n");
            }

            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("%d: SI(%d) R(%d)\n",i,sharedBitIndex[i],relation[i] );
            }
        #endif

        #pragma endregion
        //**********************************/
        //   Source backward traverse
        //**********************************/
        #pragma region sourceDistAccumulation_pushBased
            // Backward phase
            // for (int d = level - 1; d > 0; --d) {
            //     for (int w : S[d]) {
            //         for (int v : Successors[w]) {
            //             delta[w] += (sigma[w] / (float)sigma[v]) * (1.0 + delta[v]);
            //         }
            //         _BCs[w] += delta[w];
            //     }
            // }
            for (int d = level - 1; d >= 0; --d) {
                for (int node : S[d]) { 
                    for(neighborIndex = _csr->csrV[node] ; neighborIndex < _csr->csrV[node + 1] ; neighborIndex ++) {
                        int neighborNodeID =  _csr->csrE[neighborIndex];

                        //1:代表我有不一樣的鄰居(edge)  0:代表node的鄰居都是一樣的顏色(Suc)
                        sameIndex[node] |= (sharedBitIndex[node]^sharedBitIndex[neighborNodeID])|(relation[node]^relation[neighborNodeID]);
                        
                        //normal backward
                        if(dist[neighborNodeID] == dist[node] + 1 )
                            delta[node] += (sigma[node] / (float)sigma[neighborNodeID]) * (1.0 + delta[neighborNodeID]);
                        
                    }
                    ///normal BC accumulation
                    if (node != sourceID) {
                        // printf("delta[%d]: (%.2f)\n",node,delta[node]);
                        _BCs[node] += delta[node];
                    }
                }
            }
        #pragma endregion //distAccumulation_pushBased
        
        // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //     printf("_BC_first[%d]: (%.2f)\n",i,_BCs[i]);
        // }
        
        #ifdef DEBUG //print SI、R、sameIndex
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("_BC[%d]: (%.2f)\n",i,_BCs[i]);
            }
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf(" sameIndex[%d]: ",i);
                printbinary(sameIndex[i],mappingCount);
                printf("\tSI[%d]: ",i);
                printbinary(sharedBitIndex[i],mappingCount);
                printf("\t R[%d]: ",i);
                printbinary(relation[i],mappingCount);
            }
        #endif

        //****************************************/
        //   N(s) forward traverse(鄰居當Source)
        //****************************************/
        //初始Ns的資訊
        if(mappingCount){
            Ns_sigma.assign(mappingCount, vector<int>(V, 0));  // Reset Ns_sigma with zeros
            Ns_dist.assign(mappingCount, vector<int>(V, -1));  // Reset Ns_dist with -1 (unvisited)
            Ns_delta.assign(mappingCount, vector<float>(V, 0.0)); // Reset Ns_delta with 0.0
            Ns_Successors.assign(mappingCount, vector<vector<int>>(V)); // Clear and reset Ns_Successors
            Q_f2_temp.assign(V, Q_struct{0, -1}); // Reset Q_f2_temp using assign
            Ns_S.assign(V, vector<Q_struct>());  // Reset Ns_S with empty vectors
        }
        

        //這部分使用 multi-source實作
        #pragma region third_part

        // printf("---------initial Queue-----------\n");
        for (int i = 0; i < mappingCount; i++) {
            int sourceNode = mapping_SI[i];
            Ns_sigma[i][sourceNode] = 1.0;
            Ns_dist[i][sourceNode]  = 0;
            Q_f1.push({(1ULL << i),sourceNode});

            //printf
            // printf("f1[%d]:[%d,",i,sourceNode);
            // printbinary((1ULL << i),mappingCount);
        }
        // printf("----------------0-----------------\n");
        
        level=0;
        // BFS forward phase
        while (!Q_f1.empty()){
            while (!Q_f1.empty()) {
                int currentNodeID   = Q_f1.front().nodeID;
                uint64_t traverse_S = Q_f1.front().traverse_S;
                Ns_S[level].push_back(Q_f1.front());
                Q_f1.pop();
                //mapping
                // Suc/edge neighbor
                for (auto multi_node = 0; multi_node < mappingCount; multi_node++) {
                    // uint64_t bit_mask=(1ULL << multi_node);
                    if (traverse_S & (1ULL << multi_node)){ //該node在這層traverse到
                        if( (sameIndex[currentNodeID] & (1ULL << multi_node)) == 0 ){ //Suc_traverse
                            Suc_times++;
                            // printf("[%d] Suc_traverse\n",  currentNodeID);
                            for (auto SucNodeInDex = 0; SucNodeInDex < Successors[currentNodeID].size(); SucNodeInDex++) {
                                int SucNodeID = Successors[currentNodeID][SucNodeInDex];

                                Ns_dist [multi_node][SucNodeID]  = Ns_dist[multi_node][currentNodeID] +1;
                                Ns_sigma[multi_node][SucNodeID] += Ns_sigma[multi_node][currentNodeID];
                                Ns_Successors[multi_node][currentNodeID].push_back(SucNodeID);
                                //push node into Q_f2_temp
                                Q_f2_temp[SucNodeID].nodeID=SucNodeID;
                                Q_f2_temp[SucNodeID].traverse_S|= (traverse_S & (1ULL << multi_node));
                                // printf("[%d] push in Suc\n",  SucNodeID);
                            }
                        }else{//edge_traverse
                            edge_times++;
                            // printf("[%d] edge_traverse\n",  currentNodeID);
                            for (auto neighborIndex = _csr->csrV[currentNodeID]; neighborIndex < _csr->csrV[currentNodeID + 1]; neighborIndex++) {
                                int neighborNodeID = _csr->csrE[neighborIndex];
                                //累加路徑數量、dist
                                if (Ns_dist[multi_node][neighborNodeID] == -1) {
                                    Ns_Successors[multi_node][currentNodeID].push_back(neighborNodeID);
                                    
                                    Ns_dist [multi_node][neighborNodeID]  = Ns_dist[multi_node][currentNodeID] + 1;
                                    Ns_sigma[multi_node][neighborNodeID] += Ns_sigma[multi_node][currentNodeID];

                                    // printf("[%d] push in edge\n",  neighborNodeID);
                                    Q_f2_temp[neighborNodeID].nodeID=neighborNodeID;
                                    Q_f2_temp[neighborNodeID].traverse_S|= (traverse_S & (1ULL << multi_node));
                                }
                                else if (Ns_dist[multi_node][neighborNodeID] == Ns_dist[multi_node][currentNodeID] + 1) {
                                    Ns_Successors[multi_node][currentNodeID].push_back(neighborNodeID);
                                    Ns_sigma[multi_node][neighborNodeID] += Ns_sigma[multi_node][currentNodeID];
                                }
                            }
                        }

                    }
                }
                
            }
            
            //在這定義Q_f2_temp(vector)搜尋V個node，把對應index的node放進Q_f2
            for(int insert_q = _csr->startNodeID ; insert_q <=_csr->endNodeID ; insert_q++ ){
                if(Q_f2_temp[insert_q].nodeID!=-1){
                    Q_f2.push(Q_f2_temp[insert_q]);
                    //print DeBUG
                    // printf("Q_f2[%d]:[%d,",insert_q, Q_f2_temp[insert_q].nodeID);
                    // printbinary(Q_f2_temp[insert_q].traverse_S,mappingCount);
                }
            }
            // printf("----------------%d----------------\n",(level+1));
            
            //做Q_f1(空的), Q_f2交換
            swap(Q_f1, Q_f2);
            Q_f2_temp.assign(V, Q_struct{0, -1}); // Reset Q_f2_temp using assign
            level++;
        }

        //print Ns dist path Suc
        // for(int i=_csr->startNodeID; i<=_csr->endNodeID;i++){
        //     printf("dist{%d}: ",i);
        //     for(int j=0; j<Ns_dist[i].size();j++){
        //         printf("%d ",Ns_dist[i][j]);
        //     }
        //     printf("\n");
        // }
        
        // for(int i=_csr->startNodeID; i<=_csr->endNodeID;i++){
        //     printf("sigma{%d}: ",i);
        //     for(int j=0; j<Ns_sigma[i].size();j++){
        //         printf("%d ",Ns_sigma[i][j]);
        //     }
        //     printf("\n");
        // }

        // for(int i=0; i<mappingCount;i++){
        //     printf("Successors{%d}: \n",mapping_SI[i]);
        //     for(int j=_csr->startNodeID; j<=_csr->endNodeID;j++){
        //         printf("{%d}: ",j);
        //         for(auto& x:Ns_Successors[i][j]){
        //             printf("%d ",x);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }


        #pragma endregion //neighborOfSource_GetDist

        //****************************************/
        //   N(s) backward traverse
        //****************************************/
        #pragma region forth_part
        for (int d = level - 1; d > 0; --d) {
            // printf("--------back level: %d------------\n",d);
            for (auto node : Ns_S[d]) {
                for(auto multi_node = 0; multi_node < mappingCount; multi_node++){
                    uint64_t bit_mask = (1ULL<<multi_node);
                    if(node.traverse_S & bit_mask){
                        for (auto SucNodeInDex = 0; SucNodeInDex < Ns_Successors[multi_node][node.nodeID].size(); SucNodeInDex++) {
                            int SucNodeID = Ns_Successors[multi_node][node.nodeID][SucNodeInDex];
                            Ns_delta[multi_node][node.nodeID] += (Ns_sigma[multi_node][node.nodeID] / (float)Ns_sigma[multi_node][SucNodeID]) * (1.0 + Ns_delta[multi_node][SucNodeID]);
                        }
                        
                        // if(node.nodeID != mapping_SI[multi_node]){
                            // printf("Ns_delta[%d][%d]: (%.2f)\n",node.nodeID,mapping_SI[multi_node],Ns_delta[node.nodeID][multi_node]);
                            _BCs[node.nodeID] += Ns_delta[multi_node][node.nodeID];
                        // }
                        
                    }
                }
                
            }
        }

       
        #pragma endregion //neighborOfSource_GetDist

        //reset the SI & relation arrays
        relation.assign(_csr->csrVSize, 0);       // Reset relation to size _csr->csrVSize with all values 0
        sharedBitIndex.assign(_csr->csrVSize, 0); // Reset sharedBitIndex to size _csr->csrVSize with all values 0
        sameIndex.assign(_csr->csrVSize, 0);      // Reset sameIndex to size _csr->csrVSize with all values 0

    }

    unsigned long total_times=edge_times+Suc_times;
    printf("suc  traverse: %0.2f\n", Suc_times/(float)total_times);
    printf("edge traverse: %0.2f\n", edge_times/(float)total_times);
    
    // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
    //     printf("_BC[%d]: (%.2f)\n",i,_BCs[i]);
    // }

    // printf("\n[BC_sharedBased] Done!\n");
}

 #pragma region brandes
//使用shared 和successor來BC (循序版本，Ns分別traverse)
void computeBC_shareBased_Successor_SS( CSR* _csr, float* _BCs){
    //時間量測
    double time_sort=0.0;
    double time_phase1=0.0;
    double time_phase2=0.0;
    double time_phase3=0.0;
    double time_phase4=0.0;
    double time_phase4_1=0.0;
    double start_time=0.0;
    double end_time=0.0;


    // multi_time1 = seconds();
    start_time=seconds();
    // showCSR(_csr);
    int V =  _csr->csrVSize;


    //record that nodes which haven't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);
    
    //record nodes belongs to which neighbor of source
    int  mappingCount_max           = 32; //最大可以設32
    int* mapping_SI                 = (int*)malloc(sizeof(int) * mappingCount_max);
    vector<unsigned int> sharedBitIndex(_csr->csrVSize, 0); // for recording blue edge bitIndex
    vector<unsigned int> relation(_csr->csrVSize, 0);       // for recording red edge bitIndex
    vector<unsigned int> sameIndex(_csr->csrVSize, 0);// for recording blue edge BC_sameIndex

    // Allocate memory for BFS data structures 
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);               // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<vector<int>> Successors(V);     // Successors list 
    // vector<float> _BCs_check(V, 0.0);           // Delta array
    //可重複使用的空間
    vector<vector<int>> S(V);              // S is a 2D stack
    queue<int> f1, f2;                     // Current and Next frontier
    //N(s)鄰居的traverse 路徑數量、距離、Suc、Sigma
    vector<int>  Ns_dist  (V, -1);            // Distance array
    vector<int>  Ns_sigma (V, 0);             // Sigma array
    vector<float> Ns_delta(V, 0.0);           // Delta array
    vector<vector<int>> Ns_Successors(V);     // Successors [][]: [V] [V的Suc] 
    
    //Suc與edge數據
    unsigned long Suc_times=0;
    unsigned long edge_times=0;
    long S_node = 0;
    long Ns_node = 0;


    
    
    //用degree做排序 大->小
    _csr->orderedCsrV  = (int*)calloc(sizeof(int), (_csr->csrVSize) *2);
    for(int i=_csr->startNodeID;i<=_csr->endNodeID;i++){
            _csr->orderedCsrV[i]=i;
    }
    quicksort_nodeID_with_degree(_csr->orderedCsrV, _csr->csrNodesDegree, _csr->startNodeID, _csr->endNodeID);
    
    // printf("orderedCsrV: ");
    // for(int i=_csr->startNodeID;i<=_csr->endNodeID;i++){
    //         printf("[%d][%d] ",i,_csr->orderedCsrV[i]);
    // }
    // printf("\n");

    end_time = seconds();
    time_sort += end_time - start_time;
    // for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
    


    for(int sourceIDIndex = _csr->startNodeID ; sourceIDIndex <= _csr->endNodeID ; sourceIDIndex ++){
        
        int sourceID = _csr->orderedCsrV[sourceIDIndex];
        
        if(nodeDone[sourceID] == 1){
            continue;
        }
        nodeDone[sourceID] = 1;
        
        start_time=seconds();
        // printf("SourceID = %2d\n", sourceID);
        S_node++;

        int currentNodeID=-1;
        int neighborNodeID=-1;
        int neighborIndex =-1;
        //each neighbor of sourceID mapping to bit_SI, if it haven't been source yet
        //挑選鄰居
        int mappingCount = 0;
        for(neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            neighborNodeID = _csr->csrE[neighborIndex];

            if(nodeDone[neighborNodeID] == 0){
                //Record to 32 bit only
                if(mappingCount == mappingCount_max){
                    break;
                }

                sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                mapping_SI[mappingCount] = neighborNodeID;
                nodeDone[neighborNodeID] = 1;

                // printf("sharedBitIndex[%6d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #ifdef DEBUG
                printf("sharedBitIndex[%2d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #endif
                
                mappingCount ++;
            }
        }
        Ns_node+= (mappingCount);
        //**********************************/
        //   Source forward traverse
        //**********************************/
        #pragma region SourceTraverse //forward traverse
        
        // Initialize arrays for each source node
        // sigma.assign(V, 0);   // Reset sigma to size V with all values 0
        // dist.assign(V, -1);   // Reset dist to size V with all values -1
        // delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
        for (int i = 0; i < V; ++i) {
            sigma[i] = 0;   // 每個 sigma 初始化為 0
        }
        for (int i = 0; i < V; ++i) {
            dist[i] = -1;   // 每個距離初始化為 -1
        }
        for (int i = 0; i < V; ++i) {
            delta[i] = 0.0; // 每個 delta 初始化為 0.0
        }
        
        // fill(sigma.begin(), sigma.end(), 0);
        // fill(dist.begin(), dist.end(), -1);
        // fill(delta.begin(), delta.end(), 0.0);
        // S.assign(V, vector<int>());  // Reset S with empty vectors
        // Successors.assign(V, vector<int>());  // Reset Successors with empty vectors
        for (auto& level : S) {
            level.clear();
        }
        for (auto& preds : Successors) {
            preds.clear();
        }

        sigma[sourceID] = 1;
        dist[sourceID] = 0;
        f1.push(sourceID);
        int level = 0;

        // cout<<"source ID: "<<sourceID<<endl;
        // BFS forward phase
        
        while (!f1.empty()){
            while (!f1.empty()) {
                currentNodeID = f1.front();
                f1.pop();
                S[level].push_back(currentNodeID);
                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d\n", currentNodeID, dist[currentNodeID]);
                #endif
                // Traverse neighbors in CSR
                for(neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++) {
                    neighborNodeID =  _csr->csrE[neighborIndex];

                    if (dist[neighborNodeID] < 0 ) {
                        dist[neighborNodeID] = dist[currentNodeID] + 1;    
                        // sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        f2.push(neighborNodeID);
                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist[neighborNodeID]);
                        #endif
                    }
                    
                    if (dist[neighborNodeID] == dist[currentNodeID] + 1) {
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        Successors[currentNodeID].push_back(neighborNodeID);
                        sigma[neighborNodeID] += sigma[currentNodeID];
                        #ifdef DEBUG
                        printf("\t[2]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist[neighborNodeID]);
                        #endif
                    } //&& currentNodeID < neighborNodeID
                    else if(dist[neighborNodeID] == dist[currentNodeID] ){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID]     |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                        // relation[neighborNodeID]    |= sharedBitIndex[currentNodeID]  & (~sharedBitIndex[neighborNodeID]);
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif

                    }//&& relation[neighborNodeID]
                    else if(dist[neighborNodeID] == dist[currentNodeID] - 1){
                        //這一步是讓SI 跟R只會顯示一個 SI:1 R:0 幫助最終判鄰居是否相同 
                        relation[currentNodeID]     = (relation[currentNodeID]|relation[neighborNodeID]) & (~sharedBitIndex[currentNodeID] );
                        #ifdef DEBUG
                        printf("\t[4]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif
                    }

                }

            }
            swap(f1, f2);
            level++;
        }



        #ifdef DEBUG
            //printf successor
            printf("\n");
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("Successor(%d) sigma(%d) ",i,sigma[i] );
                for (int j=0;j<Successors[i].size();j++){
                    printf("%d ",Successors[i][j]);
                }
                printf("\n");
            }

            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("%d: SI(%d) R(%d)\n",i,sharedBitIndex[i],relation[i] );
            }
        #endif
        //  printf("------------Source %d--------------\n",sourceID);
        #pragma endregion

        end_time = seconds();
        time_phase1 += end_time - start_time;

        start_time = seconds();
        //**********************************/
        //   Source backward traverse
        //**********************************/
        #pragma region sourceDistAccumulation_pushBased
            // Backward phase
            // for (int d = level - 1; d > 0; --d) {
            //     for (int w : S[d]) {
            //         for (int v : Successors[w]) {
            //             delta[w] += (sigma[w] / (float)sigma[v]) * (1.0 + delta[v]);
            //         }
            //         _BCs[w] += delta[w];
            //     }
            // }
            //d >= 0 這裡一定要有 =0 為了記錄sameindex
            float ratio = 0.0f;
            for (int d = level - 1; d >= 0; --d) {
                for (int node : S[d]) {
                    // for (auto SucNodeInDex = 0; SucNodeInDex < Successors[node].size(); SucNodeInDex++) {
                    //     int SucNodeID = Successors[node][SucNodeInDex];
                    //     delta[node] += (sigma[node] / (float)sigma[SucNodeID]) * (1.0 + delta[SucNodeID]);
                    // } 
                    for(neighborIndex = _csr->csrV[node] ; neighborIndex < _csr->csrV[node + 1] ; neighborIndex ++) {
                        neighborNodeID =  _csr->csrE[neighborIndex];

                        //1:代表我有不一樣的鄰居(edge)  0:代表node的鄰居都是一樣的顏色(Suc)
                        // sameIndex[node] |= (sharedBitIndex[node] | relation[node]) ^ (sharedBitIndex[neighborNodeID] | relation[neighborNodeID]);
                        sameIndex[node] |= (sharedBitIndex[node]^sharedBitIndex[neighborNodeID])|(relation[node]^relation[neighborNodeID]);
                        
                        //normal backward
                        if(dist[neighborNodeID] == dist[node] + 1 ){
                            ratio = (float)sigma[node] / sigma[neighborNodeID];
                            delta[node] += ratio * (1.0f + Ns_delta[neighborNodeID]);
                        }

                    }
                    ///normal BC accumulation
                    if (node != sourceID) {
                        // printf("delta[%d]: (%.2f)\n",node,delta[node]);
                        _BCs[node] += delta[node];
                        // _BCs_check[node] += delta[node];
                    }
                }
            }
        #pragma endregion //distAccumulation_pushBased
        
        // printf("-------------sourceID %d check-------------\n",sourceID);
        // brandes_with_predecessors_dynamic_check_ans(*_csr,V,sourceID,delta);

        // printf("------------sourceID %d delta-------------\n",sourceID);
        // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //     printf("my_delta[%d]: (%.2f)\n",i,delta[i]);
        // }

        // printf("------------sourceID %d BC-------------\n",sourceID);
        //     for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //         printf("my_BC[%d]: (%.2f)\n",i,_BCs[i]);
        //     }

        // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //     printf("_BC_first[%d]: (%.2f)\n",i,_BCs[i]);
        // }
        
        #ifdef DEBUG //print SI、R、sameIndex
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("_BC[%d]: (%.2f)\n",i,_BCs[i]);
            }
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf(" sameIndex[%d]: ",i);
                printbinary(sameIndex[i],mappingCount);
                printf("\tSI[%d]: ",i);
                printbinary(sharedBitIndex[i],mappingCount);
                printf("\t R[%d]: ",i);
                printbinary(relation[i],mappingCount);
            }
        #endif

        end_time = seconds();
        time_phase2 += end_time - start_time;
        
        //****************************************/
        //   N(s) forward & backward traverse(鄰居當Source)
        //****************************************/

        //這部分使用 一個N(s)做完forward以及backward 再換下一個N(s)
        #pragma region third_and_forth_part
        //每個Ns做一次
        int sourceNode = -1;
        int SucNodeID  = -1;
        int SucNodeInDex = -1;
        for(auto Ns_index = 0 ; Ns_index < mappingCount ; Ns_index++ ){
            // start_time = seconds();
            start_time = seconds();
            sourceNode = mapping_SI[Ns_index];
            uint32_t Ns_mask = (1<<Ns_index);
            // printf("------------Source N(s) %d--------------\n",sourceNode);
            //****************************************/
            //               initial Ns陣列值
            //****************************************/
            //初始Ns的值
            // Ns_sigma.assign(V, 0);   // Reset sigma to size V with all values 0
            // Ns_dist.assign(V, -1);   // Reset dist to size V with all values -1
            // Ns_delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
            for (int i = 0; i < V; ++i) {
                Ns_sigma[i] = 0;   // 每個 sigma 初始化為 0
            }
            for (int i = 0; i < V; ++i) {
                Ns_dist[i] = -1;   // 每個距離初始化為 -1
            }
            for (int i = 0; i < V; ++i) {
                Ns_delta[i] = 0.0; // 每個 delta 初始化為 0.0
            }
            
            // fill(Ns_sigma.begin(), Ns_sigma.end(), 0);
            // fill(Ns_dist.begin(), Ns_dist.end(), -1);
            // fill(Ns_delta.begin(), Ns_delta.end(), 0.0);
            // S.assign(V, vector<int>());  // Reset S with empty vectors
            // Ns_Successors.assign(V, vector<int>());  // Reset Successors with empty vectors
            for (auto& level : S) {
                level.clear();
            }
            for (auto& Suc : Ns_Successors) {
                Suc.clear();
            }

            
            //****************************************/
            //               initial Source
            //****************************************/
            Ns_sigma[sourceNode] = 1;
            Ns_dist [sourceNode] = 0;
            f1.push(sourceNode);


            //****************************************/
            //         N(s) forward traverse
            //****************************************/
            level=0;
            while (!f1.empty()){
                while (!f1.empty()) {
                    int currentNodeID = f1.front();
                    S[level].push_back(currentNodeID);
                    f1.pop();

                    if( (sameIndex[currentNodeID] & Ns_mask) == 0 ){ //edge_traverse

                        Suc_times++;
                        for (  SucNodeInDex = 0 ; SucNodeInDex<Successors[currentNodeID].size(); SucNodeInDex++) {  
                            SucNodeID = Successors[currentNodeID][SucNodeInDex];
                            if(Ns_dist[SucNodeID] < 0){
                                Ns_dist [SucNodeID]  = Ns_dist[currentNodeID] + 1;
                                //push node into Q_f2_temp
                                f2.push(SucNodeID);
                            }
                            Ns_Successors[currentNodeID].push_back(SucNodeID);
                            Ns_sigma[SucNodeID] += Ns_sigma[currentNodeID];
                        }
                    }else{//Suc_traverse

                        edge_times++;
                        for (neighborIndex = _csr->csrV[currentNodeID]; neighborIndex < _csr->csrV[currentNodeID + 1]; neighborIndex++) {
                            
                            neighborNodeID = _csr->csrE[neighborIndex];
                            //累加路徑數量、dist
                            if (Ns_dist[neighborNodeID] < 0) {
                                // Ns_Successors[currentNodeID].push_back(neighborNodeID);
                                Ns_dist [neighborNodeID]  = Ns_dist[currentNodeID] + 1;
                                // Ns_sigma[neighborNodeID] += Ns_sigma[currentNodeID];
                                // printf("[%d] push in edge\n",  neighborNodeID);
                                f2.push(neighborNodeID);
                            }
                            
                            if (Ns_dist[neighborNodeID] == Ns_dist[currentNodeID] + 1) {
                                Ns_Successors[currentNodeID].push_back(neighborNodeID);
                                Ns_sigma[neighborNodeID] += Ns_sigma[currentNodeID];
                            }
                        }
                    }

                }
                swap(f1,f2);
                level++;
            }

            
            end_time = seconds();
            time_phase3 += end_time - start_time;
            start_time = seconds();
            //****************************************/
            //         N(s) backward traverse
            //****************************************/

            float ratio = 0.0f;
            for (int d = level - 1; d > 0; d--) {
                
                for (int node : S[d]) {
                    // printf("back node: %d\n",node);
                    // if( (sameIndex[node] & Ns_mask) == 0 ) { //Suc_traverse
                    //     for (int SucNodeID : Successors[node]) {
                    //         Ns_delta[node] += (Ns_sigma[node] / static_cast<float>(Ns_sigma[SucNodeID])) * (1.0 + Ns_delta[SucNodeID]);
                    //     }
                    // }else{
                        for (int SucNodeID : Ns_Successors[node]) {
                            ratio = (float)Ns_sigma[node] / Ns_sigma[SucNodeID];
                            Ns_delta[node] += ratio * (1.0f + Ns_delta[SucNodeID]);
                        }
                    // }
                    
                    
                    // if(node != sourceNode){ 
                        _BCs[node] += Ns_delta[node];
                    // } 
                }
                
            }

            // printf("-------------Ns sourceID %d check-------------\n",sourceNode);
            // brandes_with_predecessors_dynamic_check_ans(*_csr,V,sourceNode,Ns_delta);


            // printf("------------Ns sourceID %d delta-------------\n",sourceNode);
            // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
            //     printf("my_delta[%d]: (%.2f)\n",i,Ns_delta[i]);
            // }

            // printf("-------------Ns sourceID %d BC-------------\n",sourceNode);
            // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
            //     printf("my_BC[%d]: (%.2f)\n",i,_BCs[i]);
            // }
            end_time = seconds();
            time_phase4 += end_time - start_time;
           
            #ifdef DEBUG
            for(int i= _csr->startNodeID ; i<= _csr->endNodeID ;i++){
                printf("%d of Ns_delta[%d]: (%.2f)\n",sourceNode,i,Ns_delta[i]);
            }
            #endif


        }

        #pragma endregion //neighborOfSource_GetDist


        //reset the SI & relation arrays
        relation.assign(V, 0);       // Reset relation to size _csr->csrVSize with all values 0
        sharedBitIndex.assign(V, 0); // Reset sharedBitIndex to size _csr->csrVSize with all values 0
        sameIndex.assign(V, 0);      // Reset sameIndex to size _csr->csrVSize with all values 0


    }
    mymethod_time2 += mymethod_time1 + time_phase1 +time_phase2+time_phase3+time_phase4;
    // multi_time2 = seconds();
    unsigned long total_times=edge_times+Suc_times;
    printf("suc  traverse: %0.2f\n", Suc_times/(float)total_times);
    printf("edge traverse: %0.2f\n", edge_times/(float)total_times);
    printf("S_node       : %ld\n", S_node);
    printf("Ns_node      : %ld\n", Ns_node);
    printf("sort   time: %0.6f\n", time_sort);
    printf("phase1 time: %0.6f\n", time_phase1);
    printf("phase2 time: %0.6f\n", time_phase2);
    printf("phase3 time: %0.6f\n", time_phase3);
    printf("phase4 time: %0.6f\n", time_phase4);
    // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
    //     printf("my_BC[%d]: (%.2f)\n",i,_BCs[i]);
    // }

    // printf("\n[BC_sharedBased] Done!\n");
}

 #pragma endregion
void computeBC_shareBased_Successor_SS_edge_update( CSR* _csr, float* _BCs){
    //時間量測
    double time_sort=0.0;
    double time_phase1=0.0;
    double time_phase2=0.0;
    double time_phase3=0.0;
    double time_phase4=0.0;
    double start_time=0.0;
    double end_time=0.0;


    // multi_time1 = seconds();
    start_time=seconds();
    // showCSR(_csr);
    int V =  _csr->csrVSize;


    //record that nodes which haven't been source yet
    int* nodeDone = (int*)calloc(sizeof(int), _csr->csrVSize);
    
    //record nodes belongs to which neighbor of source
    int  mappingCount_max           = 32; //最大可以設32
    int* mapping_SI                 = (int*)malloc(sizeof(int) * 32);
    vector<unsigned int> sharedBitIndex(_csr->csrVSize, 0); // for recording blue edge bitIndex
    vector<unsigned int> relation(_csr->csrVSize, 0);       // for recording red edge bitIndex
    vector<unsigned int> sameIndex(_csr->csrVSize, 0);// for recording blue edge BC_sameIndex

    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);              // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);               // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<vector<int>> Successors(V);     // Successors list 
    vector<float> _BCs_check(V, 0.0);           // Delta array
    //可重複使用的空間
    queue<int> f1, f2;                     // Current and Next frontier
    //N(s)鄰居的traverse 路徑數量、距離、Suc、Sigma
    vector<vector<int>> N_S(V);              // S is a 2D stack
    vector<int>  Ns_dist  (V, -1);            // Distance array
    vector<int>  Ns_sigma (V, 0);             // Sigma array
    vector<float> Ns_delta(V, 0.0);           // Delta array
    vector<vector<int>> Ns_Successors(V);     // Successors [][]: [V] [V的Suc] 
    
    //Suc與edge數據
    unsigned long Suc_times=0;
    unsigned long edge_times=0;
    long S_node = 0;
    long Ns_node = 0;


    
    
    //用degree做排序 大->小
    _csr->orderedCsrV  = (int*)calloc(sizeof(int), (_csr->csrVSize) *2);
    for(int i=_csr->startNodeID;i<=_csr->endNodeID;i++){
            _csr->orderedCsrV[i]=i;
    }
    quicksort_nodeID_with_degree(_csr->orderedCsrV, _csr->csrNodesDegree, _csr->startNodeID, _csr->endNodeID);
    
    // printf("orderedCsrV: ");
    // for(int i=_csr->startNodeID;i<=_csr->endNodeID;i++){
    //         printf("[%d][%d] ",i,_csr->orderedCsrV[i]);
    // }
    // printf("\n");

    end_time = seconds();
    time_sort += end_time - start_time;
    // for(int sourceID = _csr->startNodeID ; sourceID <= _csr->endNodeID ; sourceID ++){
    


    for(int sourceIDIndex = _csr->startNodeID ; sourceIDIndex <= _csr->endNodeID ; sourceIDIndex ++){
        
        int sourceID = _csr->orderedCsrV[sourceIDIndex];
        
        if(nodeDone[sourceID] == 1){
            continue;
        }
        nodeDone[sourceID] = 1;
        
        start_time=seconds();
        // printf("SourceID = %2d\n", sourceID);
        // S_node++;

        
        //each neighbor of sourceID mapping to bit_SI, if it haven't been source yet
        //挑選鄰居
        int mappingCount = 0;
        for(int neighborIndex = _csr->csrV[sourceID] ; neighborIndex < _csr->csrV[sourceID + 1] ; neighborIndex ++){
            const int neighborNodeID = _csr->csrE[neighborIndex];

            if(nodeDone[neighborNodeID] == 0){
                //Record to 32 bit only
                if(mappingCount == mappingCount_max){
                    break;
                }

                sharedBitIndex[neighborNodeID] = 1 << mappingCount;
                mapping_SI[mappingCount] = neighborNodeID;
                nodeDone[neighborNodeID] = 1;

                // printf("sharedBitIndex[%6d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #ifdef DEBUG
                printf("sharedBitIndex[%2d] = %8x,\tmapping_SI[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], mappingCount, mapping_SI[mappingCount]);
                #endif
                
                mappingCount ++;
            }
        }
        // Ns_node+= (mappingCount);
        //**********************************/
        //   Source forward traverse
        //**********************************/
        #pragma region SourceTraverse //forward traverse
        
        // Initialize arrays for each source node
        sigma.assign(V, 0);   // Reset sigma to size V with all values 0
        dist.assign(V, -1);   // Reset dist to size V with all values -1
        delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
        // fill(sigma.begin(), sigma.end(), 0);
        // fill(dist.begin(), dist.end(), -1);
        // fill(delta.begin(), delta.end(), 0.0);
        // S.assign(V, vector<int>());  // Reset S with empty vectors
        // Successors.assign(V, vector<int>());  // Reset Successors with empty vectors
        for (auto& level : S) {
            level.clear();
        }
        for (auto& preds : Successors) {
            preds.clear();
        }

        sigma[sourceID] = 1;
        dist[sourceID] = 0;
        f1.push(sourceID);
        int level = 0;

        // cout<<"source ID: "<<sourceID<<endl;
        // BFS forward phase
        
        while (!f1.empty()){
            while (!f1.empty()) {
                int currentNodeID = f1.front();
                f1.pop();
                S[level].push_back(currentNodeID);
                #ifdef DEBUG
                printf("currentNodeID = %2d ... dist = %2d\n", currentNodeID, dist[currentNodeID]);
                #endif
                // Traverse neighbors in CSR
                for(int neighborIndex = _csr->csrV[currentNodeID] ; neighborIndex < _csr->csrV[currentNodeID + 1] ; neighborIndex ++) {
                    int neighborNodeID =  _csr->csrE[neighborIndex];

                    if (dist[neighborNodeID] < 0 ) {
                        dist[neighborNodeID] = dist[currentNodeID] + 1;    
                        // sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        f2.push(neighborNodeID);
                        #ifdef DEBUG
                        printf("\t[1]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist[neighborNodeID]);
                        #endif
                    }
                    
                    if (dist[neighborNodeID] == dist[currentNodeID] + 1) {
                        sharedBitIndex[neighborNodeID] |= sharedBitIndex[currentNodeID];
                        Successors[currentNodeID].push_back(neighborNodeID);
                        sigma[neighborNodeID] += sigma[currentNodeID];
                        #ifdef DEBUG
                        printf("\t[2]unvisited_SI[%2d] => %2x, dist[%2d] = %2d\n", neighborNodeID, sharedBitIndex[neighborNodeID], neighborNodeID, dist[neighborNodeID]);
                        #endif
                    } //&& currentNodeID < neighborNodeID
                    else if(dist[neighborNodeID] == dist[currentNodeID] ){ //traverse to discovered neighbor which is at same level as currentNodeID
                        relation[currentNodeID]     |= sharedBitIndex[neighborNodeID] & (~sharedBitIndex[currentNodeID]);
                        // relation[neighborNodeID]    |= sharedBitIndex[currentNodeID]  & (~sharedBitIndex[neighborNodeID]);
                        #ifdef DEBUG
                        printf("\t[3]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif

                    }//&& relation[neighborNodeID]
                    else if(dist[neighborNodeID] == dist[currentNodeID] - 1){
                        //這一步是讓SI 跟R只會顯示一個 SI:1 R:0 幫助最終判鄰居是否相同 
                        relation[currentNodeID]     = (relation[currentNodeID]|relation[neighborNodeID]) & (~sharedBitIndex[currentNodeID] );
                        #ifdef DEBUG
                        printf("\t[4]Red edge found(%2d, %2d), ", currentNodeID, neighborNodeID);
                        printf("relation[%2d] = %2x, relation[%2d] = %2x\n", currentNodeID, relation[currentNodeID], neighborNodeID, relation[neighborNodeID]);
                        #endif
                    }

                }

            }
            swap(f1, f2);
            level++;
        }



        #ifdef DEBUG
            //printf successor
            printf("\n");
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("Successor(%d) sigma(%d) ",i,sigma[i] );
                for (int j=0;j<Successors[i].size();j++){
                    printf("%d ",Successors[i][j]);
                }
                printf("\n");
            }

            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("%d: SI(%d) R(%d)\n",i,sharedBitIndex[i],relation[i] );
            }
        #endif
        //  printf("------------Source %d--------------\n",sourceID);
        #pragma endregion

        end_time = seconds();
        time_phase1 += end_time - start_time;

        start_time = seconds();
        //**********************************/
        //   Source backward traverse
        //**********************************/
        #pragma region sourceDistAccumulation_pushBased
            // Backward phase
            // for (int d = level - 1; d > 0; --d) {
            //     for (int w : S[d]) {
            //         for (int v : Successors[w]) {
            //             delta[w] += (sigma[w] / (float)sigma[v]) * (1.0 + delta[v]);
            //         }
            //         _BCs[w] += delta[w];
            //     }
            // }
            //d >= 0 這裡一定要有 =0 為了記錄sameindex
            float ratio = 0.0f;
            for (int d = level - 1; d >= 0; --d) {
                for (int node : S[d]) {
                    // for (auto SucNodeInDex = 0; SucNodeInDex < Successors[node].size(); SucNodeInDex++) {
                    //     int SucNodeID = Successors[node][SucNodeInDex];
                    //     delta[node] += (sigma[node] / (float)sigma[SucNodeID]) * (1.0 + delta[SucNodeID]);
                    // } 
                    for(int neighborIndex = _csr->csrV[node] ; neighborIndex < _csr->csrV[node + 1] ; neighborIndex ++) {
                        int neighborNodeID =  _csr->csrE[neighborIndex];

                        //1:代表我有不一樣的鄰居(edge)  0:代表node的鄰居都是一樣的顏色(Suc)
                        // sameIndex[node] |= (sharedBitIndex[node] | relation[node]) ^ (sharedBitIndex[neighborNodeID] | relation[neighborNodeID]);
                        sameIndex[node] |= (sharedBitIndex[node]^sharedBitIndex[neighborNodeID])|(relation[node]^relation[neighborNodeID]);
                        
                        //normal backward
                        if(dist[neighborNodeID] == dist[node] + 1 ){
                            ratio = (float)sigma[node] / sigma[neighborNodeID];
                            delta[node] += ratio * (1.0f + Ns_delta[neighborNodeID]);
                        }

                    }
                    ///normal BC accumulation
                    if (node != sourceID) {
                        // printf("delta[%d]: (%.2f)\n",node,delta[node]);
                        _BCs[node] += delta[node];
                        // _BCs_check[node] += delta[node];
                    }
                }
            }
        #pragma endregion //distAccumulation_pushBased
        
        // printf("-------------sourceID %d check-------------\n",sourceID);
        // brandes_with_predecessors_dynamic_check_ans(*_csr,V,sourceID,delta);

        // printf("------------sourceID %d delta-------------\n",sourceID);
        // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //     printf("my_delta[%d]: (%.2f)\n",i,delta[i]);
        // }

        // printf("------------sourceID %d BC-------------\n",sourceID);
        //     for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //         printf("my_BC[%d]: (%.2f)\n",i,_BCs[i]);
        //     }

        // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
        //     printf("_BC_first[%d]: (%.2f)\n",i,_BCs[i]);
        // }
        
        #ifdef DEBUG //print SI、R、sameIndex
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf("_BC[%d]: (%.2f)\n",i,_BCs[i]);
            }
            for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
                printf(" sameIndex[%d]: ",i);
                printbinary(sameIndex[i],mappingCount);
                printf("\tSI[%d]: ",i);
                printbinary(sharedBitIndex[i],mappingCount);
                printf("\t R[%d]: ",i);
                printbinary(relation[i],mappingCount);
            }
        #endif

        end_time = seconds();
        time_phase2 += end_time - start_time;
        
        //****************************************/
        //   N(s) forward & backward traverse(鄰居當Source)
        //****************************************/

        //這部分使用 一個N(s)做完forward以及backward 再換下一個N(s)
        #pragma region third_and_forth_part
        //每個Ns做一次
        int sourceNode =0;
        for(auto Ns_index = 0 ; Ns_index < mappingCount ; Ns_index++ ){
            // start_time = seconds();
           start_time = seconds();
            sourceNode   = mapping_SI[Ns_index];
            uint32_t Ns_mask = (1<<Ns_index);
            // printf("------------Source N(s) %d--------------\n",sourceNode);
            //****************************************/
            //               initial Ns陣列值
            //****************************************/
            //初始Ns的值
            Ns_sigma.assign(V, 0);   // Reset sigma to size V with all values 0
            Ns_dist.assign(V, -1);   // Reset dist to size V with all values -1
            Ns_delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
            // fill(Ns_sigma.begin(), Ns_sigma.end(), 0);
            // fill(Ns_dist.begin(), Ns_dist.end(), -1);
            // fill(Ns_delta.begin(), Ns_delta.end(), 0.0);
            // S.assign(V, vector<int>());  // Reset S with empty vectors
            // Ns_Successors.assign(V, vector<int>());  // Reset Successors with empty vectors
            for (auto& level : S) {
                level.clear();
            }
            for (auto& Suc : Ns_Successors) {
                Suc.clear();
            }

            
            //****************************************/
            //               initial Source
            //****************************************/
            Ns_sigma[sourceNode] = 1;
            Ns_dist [sourceNode] = 0;
            f1.push(sourceNode);


            //****************************************/
            //         N(s) forward traverse
            //****************************************/
            level=0;
            while (!f1.empty()){
                while (!f1.empty()) {
                    int currentNodeID = f1.front();
                    S[level].push_back(currentNodeID);
                    f1.pop();

                    if( (sameIndex[currentNodeID] & Ns_mask) == 0 ){ //Suc_traverse
                        // Suc_times++;
                        for (auto SucNodeInDex = 0; SucNodeInDex < Successors[currentNodeID].size(); SucNodeInDex++) {
                            int SucNodeID = Successors[currentNodeID][SucNodeInDex];
                            if(Ns_dist[SucNodeID] < 0){
                                Ns_dist [SucNodeID]  = Ns_dist[currentNodeID] + 1;
                                //push node into Q_f2_temp
                                f2.push(SucNodeID);
                            }
                            Ns_Successors[currentNodeID].push_back(SucNodeID);
                            Ns_sigma[SucNodeID] += Ns_sigma[currentNodeID];
                        }
                    }else{//edge_traverse
                        // edge_times++;
                        for (auto neighborIndex = _csr->csrV[currentNodeID]; neighborIndex < _csr->csrV[currentNodeID + 1]; neighborIndex++) {
                            const int neighborNodeID = _csr->csrE[neighborIndex];
                            //累加路徑數量、dist
                            if (Ns_dist[neighborNodeID] < 0) {
                                // Ns_Successors[currentNodeID].push_back(neighborNodeID);
                                Ns_dist [neighborNodeID]  = Ns_dist[currentNodeID] + 1;
                                // Ns_sigma[neighborNodeID] += Ns_sigma[currentNodeID];
                                // printf("[%d] push in edge\n",  neighborNodeID);
                                f2.push(neighborNodeID);
                            }
                            
                            if (Ns_dist[neighborNodeID] == Ns_dist[currentNodeID] + 1) {
                                Ns_Successors[currentNodeID].push_back(neighborNodeID);
                                Ns_sigma[neighborNodeID] += Ns_sigma[currentNodeID];
                            }
                        }
                    }

                }
                swap(f1,f2);
                level++;
            }

            
            end_time = seconds();
            time_phase3 += end_time - start_time;
            start_time = seconds();
            //****************************************/
            //         N(s) backward traverse
            //****************************************/

            float ratio = 0.0;
            for (int d = level - 1; d > 0; d--) {
                
                for (int node : S[d]) {
                    // printf("back node: %d\n",node);
                    // if( (sameIndex[node] & Ns_mask) == 0 ) { //Suc_traverse
                    //     for (int SucNodeID : Successors[node]) {
                    //         Ns_delta[node] += (Ns_sigma[node] / static_cast<float>(Ns_sigma[SucNodeID])) * (1.0 + Ns_delta[SucNodeID]);
                    //     }
                    // }else{
                        for (int SucNodeID : Ns_Successors[node]) {
                            ratio = (float)Ns_sigma[node] / Ns_sigma[SucNodeID];
                            Ns_delta[node] += ratio * (1.0f + Ns_delta[SucNodeID]);
                        }
                    // }
                    
                    
                    // if(node != sourceNode){
                        // if(node == 5)
                        //     printf("brfore BC: (%0.2f) Ns_delta: (%.2f)\n",_BCs[node],Ns_delta[node]);
                        _BCs[node] += Ns_delta[node];
                        // if(node == 5)
                        //     printf("after BC: (%0.2f) Ns_delta: (%.2f)\n",_BCs[node],Ns_delta[node]);
                    // } 
                }
                
            }

            // printf("-------------Ns sourceID %d check-------------\n",sourceNode);
            // brandes_with_predecessors_dynamic_check_ans(*_csr,V,sourceNode,Ns_delta);


            // printf("------------Ns sourceID %d delta-------------\n",sourceNode);
            // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
            //     printf("my_delta[%d]: (%.2f)\n",i,Ns_delta[i]);
            // }

            // printf("-------------Ns sourceID %d BC-------------\n",sourceNode);
            // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
            //     printf("my_BC[%d]: (%.2f)\n",i,_BCs[i]);
            // }
            end_time = seconds();
            time_phase4 += end_time - start_time;
           
            #ifdef DEBUG
            for(int i= _csr->startNodeID ; i<= _csr->endNodeID ;i++){
                printf("%d of Ns_delta[%d]: (%.2f)\n",sourceNode,i,Ns_delta[i]);
            }
            #endif


        }

        #pragma endregion //neighborOfSource_GetDist


        //reset the SI & relation arrays
        relation.assign(V, 0);       // Reset relation to size _csr->csrVSize with all values 0
        sharedBitIndex.assign(V, 0); // Reset sharedBitIndex to size _csr->csrVSize with all values 0
        sameIndex.assign(V, 0);      // Reset sameIndex to size _csr->csrVSize with all values 0


    }

    // multi_time2 = seconds();
    unsigned long total_times=edge_times+Suc_times;
    printf("suc  traverse: %0.2f\n", Suc_times/(float)total_times);
    printf("edge traverse: %0.2f\n", edge_times/(float)total_times);
    printf("S_node       : %ld\n", S_node);
    printf("Ns_node      : %ld\n", Ns_node);
    printf("sort   time: %0.6f\n", time_sort);
    printf("phase1 time: %0.6f\n", time_phase1);
    printf("phase2 time: %0.6f\n", time_phase2);
    printf("phase3 time: %0.6f\n", time_phase3);
    printf("phase4 time: %0.6f\n", time_phase4);
    // for (int i= _csr->startNodeID;i<=_csr->endNodeID;i++) {
    //     printf("my_BC[%d]: (%.2f)\n",i,_BCs[i]);
    // }

    // printf("\n[BC_sharedBased] Done!\n");
}



void computeBC_shareBased_Successor_SS_test( CSR* csr, float* _BCs){
    // Allocate memory for BFS data structures
    // multi_time1 = seconds();
    int V = csr->csrVSize;
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);               // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> Sucecessors(V);   // Predecessor list

    // long long total_predecessor_count = 0; // To accumulate total predecessors

    double time_sort=0.0;
    double time_phase1=0.0;
    double time_phase2=0.0;
    double time_phase3=0.0;
    double time_phase4=0.0;
    double start_time=0.0;
    double end_time=0.0;

    for (int s = csr->startNodeID; s <= csr->endNodeID; ++s) {
        start_time=seconds();
        // Initialize arrays for each source node
        sigma.assign(V, 0);
        dist.assign(V, -1);
        delta.assign(V, 0);
        for (auto& level : S) {
            level.clear();
        }
        for (auto& preds : Sucecessors) {
            preds.clear();
        }

        sigma[s] = 1;
        dist[s] = 0;
        f1.push(s);

        int level = 0;

        // BFS forward phase
        while (!f1.empty()) {
            while (!f1.empty()) {
                int u = f1.front();
                f1.pop();
                S[level].push_back(u);

                // Traverse neighbors in CSR
                for (int i = csr->csrV[u]; i < csr->csrV[u + 1]; ++i) {
                    int w = csr->csrE[i];

                    if (dist[w] < 0) {
                        dist[w] = dist[u] + 1;
                        sigma[w] += sigma[u];
                        Sucecessors[u].push_back(w);
                        f2.push(w);
                    }
                    else if (dist[w] == dist[u] + 1) {
                        sigma[w] += sigma[u];
                        Sucecessors[u].push_back(w);
                    }
                }
            }
            swap(f1, f2);
            level++;
        }

        end_time=seconds();
        time_phase1 += end_time - start_time;
        start_time=seconds();
        // Backward phase
        for (int d = level - 1; d >= 0; --d) {
            for (int w : S[d]) {
                for (int v : Sucecessors[w]) {
                    delta[w] += (sigma[w] / (float)sigma[v]) * (1.0 + delta[v]);
                }
                if (w != s) {
                    _BCs[w] += delta[w];
                }
            }
        }

        end_time=seconds();
        time_phase2 += end_time - start_time;
       
    }
    // multi_time2 = seconds();
    printf("phase1 time: %0.6f\n", time_phase1);
    printf("phase2 time: %0.6f\n", time_phase2);

}

//************************************************ */
//                   循序_brandes MS原版
//************************************************ */
void Seq_multi_source_brandes( CSR& csr, int max_multi, vector<float> &BC) {
    // Start timing
    // multi_time_start = seconds();

    int v_size = csr.csrVSize;
    int* map_S = (int*)malloc(sizeof(int) * max_multi); // Multiple sources
    bool* nodeDone = (bool*)calloc(v_size, sizeof(bool));

    size_t multi_size = v_size * max_multi;

    int* s_size = (int*)malloc(sizeof(int) * v_size);
    float* dist_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* sigma_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* delta_MULTI = (float*)malloc(sizeof(float) * multi_size);

    Q_struct** s = (Q_struct**)malloc(v_size * sizeof(Q_struct*));
    Q_struct* f1 = (Q_struct*)malloc(v_size * sizeof(Q_struct));
    Q_struct* f2 = (Q_struct*)malloc(v_size * sizeof(Q_struct));

    // Pre-initialize dist_MULTI to INFINITE
    float* dist_INIT = (float*)malloc(sizeof(float) * multi_size);
    
    for (size_t i = 0; i < multi_size; i++) {
        dist_INIT[i] = INFINITE;
    }

    for (int sourceID = csr.startNodeID; sourceID <= csr.endNodeID; ++sourceID) {
        if (nodeDone[sourceID]) continue;

        // multi_time1 = seconds();

        nodeDone[sourceID] = true;
        int mappingCount = 0;
        map_S[mappingCount++] = sourceID;

        // Find other sources
        for (int neighborIndex = csr.csrV[sourceID]; neighborIndex < csr.csrV[sourceID + 1] && mappingCount < max_multi; neighborIndex++) {
            int neighborNodeID = csr.csrE[neighborIndex];
            if (!nodeDone[neighborNodeID]) {
                map_S[mappingCount++] = neighborNodeID;
                nodeDone[neighborNodeID] = true;
            }
        }

        // Initialize dist_MULTI, sigma_MULTI, delta_MULTI
        memcpy(dist_MULTI, dist_INIT, sizeof(float) * multi_size);
        memset(sigma_MULTI, 0, sizeof(float) * multi_size);
        memset(delta_MULTI, 0, sizeof(float) * multi_size);
        memset(s_size, 0, sizeof(int) * v_size);

        int f1_indicator = 0;
        int f2_indicator = 0;
        int s_indicator = 0;

        // Initialize currentQueue
        for (int i = 0; i < mappingCount; i++) {
            int sourceNode = map_S[i];
            int position = mappingCount * sourceNode + i;
            sigma_MULTI[position] = 1.0;
            dist_MULTI[position] = 0.0;

            f1[f1_indicator].nodeID = sourceNode;
            f1[f1_indicator].traverse_S = (1ULL << i);
            f1_indicator++;
        }

        // Initialize nextQueue traverse_S to zero
        for (int i = 0; i < v_size; i++) {
            f2[i].traverse_S = 0;
        }

        int level=0;
        while (f1_indicator > 0) {

            Q_struct* currentQueue;
            Q_struct* nextQueue;
            if(level% 2 == 0){
                currentQueue = f1;
                nextQueue = f2;
            }
            else{
                currentQueue = f2;
                nextQueue = f1;
            }
            // Store currentQueue into s[s_indicator]
            s[s_indicator] = (Q_struct*)malloc(f1_indicator * sizeof(Q_struct));
            memcpy(s[s_indicator], currentQueue, f1_indicator * sizeof(Q_struct));
            s_size[s_indicator++] = f1_indicator;

            // Process currentQueue
            for (auto i = 0; i < f1_indicator; i++) {
                int v = currentQueue[i].nodeID;
                uint64_t traverse_S = currentQueue[i].traverse_S;

                for (auto neighborIndex = csr.csrV[v]; neighborIndex < csr.csrV[v + 1]; neighborIndex++) {
                    int neighborNodeID = csr.csrE[neighborIndex];

                    for (auto multi_node = 0; multi_node < mappingCount; multi_node++) {
                        if (traverse_S & (1ULL << multi_node)) {
                            int position_v = mappingCount * v + multi_node;
                            int position_n = mappingCount * neighborNodeID + multi_node;

                            if (dist_MULTI[position_n] == dist_MULTI[position_v] + 1) {
                                sigma_MULTI[position_n] += sigma_MULTI[position_v];
                            } else if (dist_MULTI[position_n] > dist_MULTI[position_v] + 1) {
                                dist_MULTI[position_n] = dist_MULTI[position_v] + 1;
                                sigma_MULTI[position_n] = sigma_MULTI[position_v];

                                // Check if neighborNodeID is already in nextQueue
                                int found = -1;
                                for (auto find = 0; find < f2_indicator; find++) {
                                    if (nextQueue[find].nodeID == neighborNodeID) {
                                        found = find;
                                        break;
                                    }
                                }
                                if (found >= 0) {
                                    nextQueue[found].traverse_S |= (1ULL << multi_node);
                                } else {
                                    nextQueue[f2_indicator].nodeID = neighborNodeID;
                                    nextQueue[f2_indicator].traverse_S = (1ULL << multi_node);
                                    f2_indicator++;
                                }
                            }
                        }
                    }
                }
            }

            // Swap currentQueue and nextQueue
            f1_indicator = f2_indicator;
            f2_indicator = 0;
            level++;
            // Reset nextQueue traverse_S for next iteration
            // for (int i = 0; i < f1_indicator; i++) {
            //     nextQueue[i].traverse_S = 0;
            // }
        }

        // multi_time2 = seconds();
        // multi_forward_Time += (multi_time2 - multi_time1);
        // multi_time1 = seconds();

        // Back-propagation
        for (int layer = s_indicator - 1; layer >= 0; layer--) {
            for (int i = 0; i < s_size[layer]; i++) {
                int v = s[layer][i].nodeID;
                uint64_t traverse_S = s[layer][i].traverse_S;

                for (int multi_node = 0; multi_node < mappingCount; multi_node++) {
                    if (traverse_S & (1ULL << multi_node)) {
                        int position_v = mappingCount * v + multi_node;

                        float coeff = 0.0;

                        // For each neighbor w of v
                        for (int neighborIndex = csr.csrV[v]; neighborIndex < csr.csrV[v + 1]; neighborIndex++) {
                            int w = csr.csrE[neighborIndex];
                            int position_w = mappingCount * w + multi_node;

                            if (dist_MULTI[position_w] == dist_MULTI[position_v] + 1) {
                                coeff += (sigma_MULTI[position_v] / sigma_MULTI[position_w]) * (1.0 + delta_MULTI[position_w]);
                            }
                        }

                        delta_MULTI[position_v] += coeff;

                        // If v is not the source node, accumulate delta into BC[v]
                        if (v != map_S[multi_node]) {
                            BC[v] += delta_MULTI[position_v];
                        }
                    }
                }
            }
            // Free s[layer]
            free(s[layer]);
        }

        // multi_time2 = seconds();
        // multi_backward_Time += (multi_time2 - multi_time1);
    }

    // multi_time_end = seconds();
    // multi_total_time = (multi_time_end - multi_time_start);

    // Free memory
    free(s_size);
    free(dist_MULTI);
    free(sigma_MULTI);
    free(delta_MULTI);
    free(map_S);
    free(nodeDone);
    free(s);
    free(f1);
    free(f2);
    free(dist_INIT);
}

void Seq_multi_source_brandes_ordered( CSR& csr, int max_multi, vector<float> &BC) {
    // Start timing
    // multi_time_start = seconds();

    int v_size = csr.csrVSize;
    int* map_S = (int*)malloc(sizeof(int) * max_multi); // Multiple sources
    bool* nodeDone = (bool*)calloc(v_size, sizeof(bool));

    size_t multi_size = v_size * max_multi;

    int* s_size = (int*)malloc(sizeof(int) * v_size);
    float* dist_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* sigma_MULTI = (float*)malloc(sizeof(float) * multi_size);
    float* delta_MULTI = (float*)malloc(sizeof(float) * multi_size);

    Q_struct** s = (Q_struct**)malloc(v_size * sizeof(Q_struct*));
    Q_struct* f1 = (Q_struct*)malloc(v_size * sizeof(Q_struct));
    Q_struct* f2 = (Q_struct*)malloc(v_size * sizeof(Q_struct));

    // Pre-initialize dist_MULTI to INFINITE
    float* dist_INIT = (float*)malloc(sizeof(float) * multi_size);
    
    for (size_t i = 0; i < multi_size; i++) {
        dist_INIT[i] = INFINITE;
    }

    //order ID by degree
    csr.orderedCsrV  = (int*)calloc(sizeof(int), (csr.csrVSize) *2);
    for(int i=csr.startNodeID;i<=csr.endNodeID;i++){
            csr.orderedCsrV[i]=i;
    }
    quicksort_nodeID_with_degree(csr.orderedCsrV, csr.csrNodesDegree, csr.startNodeID, csr.endNodeID);
    // cout<<"after sort\n";
    // for(int i=csr.startNodeID;i<=csr.endNodeID;i++){
    //     cout<<csr.orderedCsrV[i]<<"[ "<<csr.csrNodesDegree[csr.orderedCsrV[i]]<<" ]>";
    // }
    // cout<<endl;

    for (int sourceIndex = csr.startNodeID; sourceIndex <= csr.endNodeID; ++sourceIndex) {
        int sourceID =csr.orderedCsrV[sourceIndex];
        if (nodeDone[sourceID]) continue;

        // multi_time1 = seconds();

        nodeDone[sourceID] = true;
        int mappingCount = 0;
        map_S[mappingCount++] = sourceID;

        // Find other sources
        for (int neighborIndex = csr.csrV[sourceID]; neighborIndex < csr.csrV[sourceID + 1] && mappingCount < max_multi; neighborIndex++) {
            int neighborNodeID = csr.csrE[neighborIndex];
            if (!nodeDone[neighborNodeID]) {
                map_S[mappingCount++] = neighborNodeID;
                nodeDone[neighborNodeID] = true;
            }
        }

        // Initialize dist_MULTI, sigma_MULTI, delta_MULTI
        memcpy(dist_MULTI, dist_INIT, sizeof(float) * multi_size);
        memset(sigma_MULTI, 0, sizeof(float) * multi_size);
        memset(delta_MULTI, 0, sizeof(float) * multi_size);
        memset(s_size, 0, sizeof(int) * v_size);

        int f1_indicator = 0;
        int f2_indicator = 0;
        int s_indicator = 0;

        // Initialize currentQueue
        for (int i = 0; i < mappingCount; i++) {
            int sourceNode = map_S[i];
            int position = mappingCount * sourceNode + i;
            sigma_MULTI[position] = 1.0;
            dist_MULTI[position] = 0.0;

            f1[f1_indicator].nodeID = sourceNode;
            f1[f1_indicator].traverse_S = (1ULL << i);
            f1_indicator++;
        }

        // Initialize nextQueue traverse_S to zero
        for (int i = 0; i < v_size; i++) {
            f2[i].traverse_S = 0;
        }

        int level=0;
        while (f1_indicator > 0) {

            Q_struct* currentQueue;
            Q_struct* nextQueue;
            if(level% 2 == 0){
                currentQueue = f1;
                nextQueue = f2;
            }
            else{
                currentQueue = f2;
                nextQueue = f1;
            }
            // Store currentQueue into s[s_indicator]
            s[s_indicator] = (Q_struct*)malloc(f1_indicator * sizeof(Q_struct));
            memcpy(s[s_indicator], currentQueue, f1_indicator * sizeof(Q_struct));
            s_size[s_indicator++] = f1_indicator;

            // Process currentQueue
            for (auto i = 0; i < f1_indicator; i++) {
                int v = currentQueue[i].nodeID;
                uint64_t traverse_S = currentQueue[i].traverse_S;

                for (auto neighborIndex = csr.csrV[v]; neighborIndex < csr.csrV[v + 1]; neighborIndex++) {
                    int neighborNodeID = csr.csrE[neighborIndex];

                    for (auto multi_node = 0; multi_node < mappingCount; multi_node++) {
                        if (traverse_S & (1ULL << multi_node)) {
                            int position_v = mappingCount * v + multi_node;
                            int position_n = mappingCount * neighborNodeID + multi_node;

                            if (dist_MULTI[position_n] == dist_MULTI[position_v] + 1) {
                                sigma_MULTI[position_n] += sigma_MULTI[position_v];
                            } else if (dist_MULTI[position_n] > dist_MULTI[position_v] + 1) {
                                dist_MULTI[position_n] = dist_MULTI[position_v] + 1;
                                sigma_MULTI[position_n] = sigma_MULTI[position_v];

                                // Check if neighborNodeID is already in nextQueue
                                int found = -1;
                                for (auto find = 0; find < f2_indicator; find++) {
                                    if (nextQueue[find].nodeID == neighborNodeID) {
                                        found = find;
                                        break;
                                    }
                                }
                                if (found >= 0) {
                                    nextQueue[found].traverse_S |= (1ULL << multi_node);
                                } else {
                                    nextQueue[f2_indicator].nodeID = neighborNodeID;
                                    nextQueue[f2_indicator].traverse_S = (1ULL << multi_node);
                                    f2_indicator++;
                                }
                            }
                        }
                    }
                }
            }

            // Swap currentQueue and nextQueue
            f1_indicator = f2_indicator;
            f2_indicator = 0;
            level++;
            // Reset nextQueue traverse_S for next iteration
            // for (int i = 0; i < f1_indicator; i++) {
            //     nextQueue[i].traverse_S = 0;
            // }
        }

        // multi_time2 = seconds();
        // multi_forward_Time += (multi_time2 - multi_time1);
        // multi_time1 = seconds();

        // Back-propagation
        for (int layer = s_indicator - 1; layer >= 0; layer--) {
            for (int i = 0; i < s_size[layer]; i++) {
                int v = s[layer][i].nodeID;
                uint64_t traverse_S = s[layer][i].traverse_S;

                for (int multi_node = 0; multi_node < mappingCount; multi_node++) {
                    if (traverse_S & (1ULL << multi_node)) {
                        int position_v = mappingCount * v + multi_node;

                        float coeff = 0.0;

                        // For each neighbor w of v
                        for (int neighborIndex = csr.csrV[v]; neighborIndex < csr.csrV[v + 1]; neighborIndex++) {
                            int w = csr.csrE[neighborIndex];
                            int position_w = mappingCount * w + multi_node;

                            if (dist_MULTI[position_w] == dist_MULTI[position_v] + 1) {
                                coeff += (sigma_MULTI[position_v] / sigma_MULTI[position_w]) * (1.0 + delta_MULTI[position_w]);
                            }
                        }

                        delta_MULTI[position_v] += coeff;

                        // If v is not the source node, accumulate delta into BC[v]
                        if (v != map_S[multi_node]) {
                            BC[v] += delta_MULTI[position_v];
                        }
                    }
                }
            }
            // Free s[layer]
            free(s[layer]);
        }

        // multi_time2 = seconds();
        // multi_backward_Time += (multi_time2 - multi_time1);
    }

    // multi_time_end = seconds();
    // multi_total_time = (multi_time_end - multi_time_start);

    // Free memory
    free(s_size);
    free(dist_MULTI);
    free(sigma_MULTI);
    free(delta_MULTI);
    free(map_S);
    free(nodeDone);
    free(s);
    free(f1);
    free(f2);
    free(dist_INIT);
}



//************************************************ */
//                   平行程式 SS
//************************************************ */

__global__ void resetBC_value(float* dist,int* f1,int* sigma,float* delta,int* stack,int* level,int target,int size){

    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < size){
        dist[idx] = 1<<20;
        sigma[idx] = 0;
        delta[idx] = 0;
        level[idx] = -1;
        f1[idx] = -1;
    }
    f1[0] = target;
    stack[0] = target;
    if(idx == target){
        dist[idx] = 0.0f;
        sigma[idx] = 1;
        level[idx] = 0;
    }
}

__global__ void printArray_int(int* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("g_dist[%d] = %d\n", idx, array[idx]);
    }
}
__global__ void printArray_float(float* array, int size,int mappingcount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("[ ");
        for(int i=0;i<mappingcount;i++){
            printf("%f,", array[idx+i]);
        }
        printf("]>");
        
    }
}
__global__ void printstack_int(int* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("stack[%d] = %d\n", idx, array[idx]);
    }
}

__device__ __forceinline__ float atomicMinFloat (float * addr, float value){
     float old;
     old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
          __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
     return old;
}


__device__  __forceinline__ void atomicOr64(uint64_t* address, uint64_t val) {
    uint32_t* address_as_ui = (uint32_t*)address;
    uint32_t old_lo, old_hi, new_lo, new_hi;

    // 循環直到成功進行原子操作
    do {
        old_lo = address_as_ui[0];
        old_hi = address_as_ui[1];
        new_lo = old_lo | (uint32_t)(val);
        new_hi = old_hi | (uint32_t)(val >> 32);
    } while ((atomicCAS(address_as_ui, old_lo, new_lo) != old_lo) || 
             (atomicCAS(address_as_ui + 1, old_hi, new_hi) != old_hi));
}

__global__ void allBC(int* g_csrV,int* g_csrE ,int* nextQueueSize,int* currentQueue,int* nextQueue,float* dist,int* sigma,int blocknum,int j,int size){

    register const int bid = blockIdx.x + j * blocknum; // 0 + 0 * INT_MAX

    if(bid > size || currentQueue[bid] == -1) return; //大於currentQueueSize

    register const int node = currentQueue[bid];
    register const int degree = g_csrV[node+1] - g_csrV[node];
    register const int threadOffset = (int)ceil(degree/(blockDim.x*1.0)); //需要看的鄰居，疊代次數
    register float     old;
    // printf("bid: %d,node: %d ,degree: %d, blockDim.x: %d\n",bid,node,degree,blockDim.x);
    for(int i=0;i<threadOffset;i++){
        register const int position = g_csrV[node] + threadIdx.x + i * blockDim.x;
        if(position < g_csrV[node+1] ){
            // printf("node: %d ,position: %d, dist: %d\n",node,g_csrE[position],dist[g_csrE[position]]);
            if(dist[node] + 1.0 < dist[g_csrE[position]]){
                //Unweighted
                // dist[g_csrE[position]] = dist[node] + 1;
                //Weighted
                old = atomicMinFloat(&dist[g_csrE[position]], (dist[node] + 1.0));
                // printf("old: %d, dist: %d\n",old,dist[g_csrE[position]]);
                if(old != dist[g_csrE[position]]){
                int next = atomicAdd(nextQueueSize,1);
                nextQueue[next] = __ldg(&g_csrE[position]);
                // printf("nextQueue[%d]: %d\n",next,nextQueue[next]);
                    // printf("%d(%d) %d(%d)\n",node,level[node],adjacencyList[position],level[adjacencyList[position]]);
                    // printf("A: %d(%.2f) --> %d(%.2f)\n",node,dist[node],adjacencyList[position],dist[adjacencyList[position]]);
                }
            }
            if(dist[node] + 1 == dist[g_csrE[position]]){
                atomicAdd(&sigma[g_csrE[position]],sigma[node]);
                //printf("B: %d(%f) --> %d(%f)\n",node,sigma[node],adjacencyList[position],sigma[adjacencyList[position]]);
            }
            // printf("node: %d ,dist: %d, sigma: %d \n",g_csrE[position],dist[g_csrE[position]],sigma[g_csrE[position]]);
        }
    }
}

__global__ void deltaCalculation(int* g_csrV,int* g_csrE,float* g_delta,int* sigma,int* stack,float* dist,int blocknum,int j,int startposition,int size){

    register const int bid = blockIdx.x + j*blocknum;
    register const int node = stack[startposition + bid];
    // printf("traverse node: %d\n",node);
    if(bid >= size || node == -1) return;

    register const int degree = g_csrV[node+1] - g_csrV[node];
    register const int threadOffset = (int)ceil(degree/(blockDim.x*1.0));

    for(int i=0;i<threadOffset;i++) {
        register const int position = g_csrV[node] + threadIdx.x + i * blockDim.x;
        if(position < g_csrV[node+1] && dist[node] - 1.0 == dist[g_csrE[position]]){
            // printf("traverse node: %d\n",node);
            atomicAdd(&g_delta[g_csrE[position]],((float)sigma[g_csrE[position]]/sigma[node])*(1.0+g_delta[node]));
            //printf("%d(%d,%.2f) %d(%d,%.2f)\n",node,level[node],sigma[node],adjacencyList[position],level[adjacencyList[position]],sigma[adjacencyList[position]]);
            // printf("g_delta[%d]: %f\n",g_csrE[position],g_delta[g_csrE[position]]);

        }
    }

}

__global__ void sum_BC_Result(float* result,float* delta,int size,int s){

    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < size && idx!=s){
        result[idx] += delta[idx];
        // printf("BC[%d]: %f\n",idx,result[idx]);
    }
        

}



void brandes_SS_par( CSR& csr, int V, float *BC) {

    //CPU variable
    int    currentQueueSize;
    int*   stackOffset = (int*)calloc(V,sizeof(int));
    //GPU MALLOC　variable
    int*   g_stack;      
    int*   g_sigma;     
    float* g_dist;
    int*   g_level;     
    float* g_delta; 
    int*   g_S_size;
    int*   g_f1;
    int*   g_f2;
    int*   g_nextQueueSize; //用來回傳給CPU判別currentQueueSize，是否繼續traverse
    int*   g_csrE;
    int*   g_csrV;
    float* g_BC;

    // printf("start malloc\n");
    cudaMalloc((void **)&g_stack,V * sizeof(int)); //用CPU的stack offset存每一層的位置
    cudaMalloc((void **)&g_sigma,V * sizeof(int));
    cudaMalloc((void **)&g_dist,V * sizeof(float));
    cudaMalloc((void **)&g_level,V * sizeof(int));
    cudaMalloc((void **)&g_delta,V * sizeof(float));
    cudaMalloc((void **)&g_S_size,V*sizeof(int));
    
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte,&total_byte);
    cudaMalloc((void **)&g_f1, free_byte * 0.3);
    cudaMalloc((void **)&g_f2, free_byte * 0.3);
    cudaMalloc((void **)&g_nextQueueSize,sizeof(int));
    cudaMalloc((void **)&g_csrV, V * sizeof(int));
    cudaMalloc((void **)&g_csrE, csr.csrESize * sizeof(int));
    cudaMalloc((void **)&g_BC, V * sizeof(float));
    cudaMemcpy(g_csrV,csr.csrV ,  V * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(g_csrE,csr.csrE ,  csr.csrESize * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(g_BC, 0.0f, V * sizeof(float));
    // printf("end malloc\n");
    // std::cout << "Total GPU memory: " << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << "Free GPU memory: " << free_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    int threadnum = 32;
   
    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        // Initialize variables for each source node
        
        //初始g_f1 queue
        resetBC_value<<<ceil(V/64.0),min(V,64)>>>(g_dist,g_f1,g_sigma,g_delta,g_stack,g_level,s,V);
        CHECK(cudaDeviceSynchronize());
        cudaMemset(g_nextQueueSize,0,sizeof(int));
        currentQueueSize = 1;
        
        int level =0;
        // BFS forward phase: frontier-based BFS with extra mallocs
        while (currentQueueSize>0) { //!qIsEmpty(current_queue)
            // printf(" forward level: %d\n",level);
            // printf("currentQueueSize: %d\n",currentQueueSize);
            // Allocate new memory for next_queue in each iteration
            int *g_currentQueue;
            int *g_nextQueue;
            if(level% 2 == 0){
                g_currentQueue = g_f1;
                g_nextQueue = g_f2;
            }
            else{
                g_currentQueue = g_f2;
                g_nextQueue = g_f1;
            }

            stackOffset[level+1] = currentQueueSize + stackOffset[level];
            int blocknum = (currentQueueSize < INT_MAX) ? currentQueueSize : INT_MAX;
            //平行跑BFS
            for(int i=0;i<(int)ceil(currentQueueSize/(float)INT_MAX);i++){
                allBC<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_nextQueueSize,g_currentQueue,g_nextQueue,g_dist,g_sigma,INT_MAX,i,currentQueueSize);
                CHECK(cudaDeviceSynchronize());
            }
                    

            
            cudaMemcpy(&currentQueueSize,g_nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(&g_stack[stackOffset[level+1]],g_nextQueue,currentQueueSize*sizeof(int),cudaMemcpyDeviceToDevice);
            cudaMemset(g_nextQueueSize,0,sizeof(int));
           
            level++;
            
        }
        // printstack_int<<<ceil(V/64.0),min(V,64)>>>(g_stack,V);
        // cudaDeviceSynchronize();
        // printArray_int<<<ceil(V/64.0),min(V,64)>>>(g_dist,V);
        // Backward phase to compute BC values
        // for(int st=0;st<V;st++){
        //     printf("stackOffset[%d]: %d\n",st,stackOffset[st]);
        // }
        // printf("total level: %d\n",level);
        for (int d = level - 1; d >= 0; --d) {
            // std::cout << "backward level(" << d << "):\t" << stackOffset[d+1] - stackOffset[d] << std::endl;
            int degree =(stackOffset[d+1] - stackOffset[d]);
            int blocknum = (degree < INT_MAX) ? degree : INT_MAX;
            
            for(int i=0;i<(int)ceil(degree/(float)INT_MAX);i++)
                deltaCalculation<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_delta,g_sigma,g_stack,g_dist,INT_MAX,i,stackOffset[d],degree);
            
            CHECK(cudaDeviceSynchronize());
            // printArray_float<<<ceil(V/64.0),min(V,64)>>>(g_delta,V);
        }
        // printArray_float<<<ceil(V/64.0),min(V,64)>>>(g_delta,V);

        sum_BC_Result<<<ceil(V/64.0),min(V,64)>>>(g_BC,g_delta,V,s);
        CHECK(cudaDeviceSynchronize());
    }
    cudaMemcpy(BC,g_BC ,  V * sizeof(int),cudaMemcpyDeviceToHost);
    // Free memory for S and its levels
    free(stackOffset);
    cudaFree(g_sigma);
    cudaFree(g_delta);
    cudaFree(g_stack);
    cudaFree(g_level);
    cudaFree(g_dist);
    cudaFree(g_f1);
    cudaFree(g_f2);
    cudaFree(g_nextQueueSize);
    cudaFree(g_BC);
}


//************************************************ */
//                   平行程式 MS
//************************************************ */


__global__ void resetBC_value_MS(float* dist,Q_struct* f1,Q_struct* f2,int* sigma,float* delta,Q_struct* stack,int target,int size){

    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < size){
        dist[idx] = 1<<20;
        sigma[idx] = 0;
        delta[idx] = 0.0f;
        f1[idx].nodeID = -1;
        f1[idx].traverse_S=0;
        f2[idx].nodeID = -1;
        f2[idx].traverse_S=0;
    }
    // f1[0].nodeID = target;
    // stack[0].nodeID = target;
    // if(idx == target){
    //     dist[idx] = 0.0f;
    //     sigma[idx] = 1;
    // }
}

__global__ void INITIAL_value_MS(float* dist,Q_struct* f1,int* sigma,int* g_map_S,int size){

    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < size){
        register const int sourceID = g_map_S[idx];
        register const int position = size * sourceID + idx; //size:mapping count
        // printf("position: %d idx: %d\n",position,idx);
        sigma[position]=1;
        dist[position]=0.0f;
        f1[idx].nodeID = sourceID;
        f1[idx].traverse_S=1ULL << idx;
    }
    // f1[0].nodeID = target;
    // stack[0].nodeID = target;
    // if(idx == target){
    //     dist[idx] = 0.0f;
    //     sigma[idx] = 1;
    // }
}

__global__ void deltaCalculation_MS(int* g_csrV,int* g_csrE,float* g_delta,int* sigma,Q_struct* stack,float* dist,int blocknum,int j,int startposition,int size,int mappingcount){

    register const int bid = blockIdx.x + j*blocknum;
    register const int node = stack[startposition + bid].nodeID;
    register const uint64_t traverse_S = stack[startposition + bid].traverse_S;
    // printf("traverse node: %d\n",node);
    if(bid >= size || node == -1) return;

    register const int degree = g_csrV[node+1] - g_csrV[node];
    register const int threadOffset = (int)ceil(degree/(blockDim.x*1.0));

    for(int i=0;i<threadOffset;i++) {
        register const int position = g_csrV[node] + threadIdx.x + i * blockDim.x;
        if(position < g_csrV[node+1]){ //&& dist[node] - 1 == dist[g_csrE[position]]
            // printf("traverse node: %d\n",node);
            register const int neighborNodeID = g_csrE[position];
            for (int multi_node = 0; multi_node < mappingcount; multi_node++) {
                // printf("multi_node: %d \n",multi_node);
                if (traverse_S & (1ULL << multi_node)) {
                    register const int position_v = mappingcount * node           + multi_node;
                    register const int position_n = mappingcount * neighborNodeID + multi_node;
                    if(dist[position_v] - 1.0 == dist[position_n]){
                        atomicAdd(&g_delta[position_n],((float)sigma[position_n]/sigma[position_v])*(1.0+g_delta[position_v]));
                    }
                }
            }  
            //printf("%d(%d,%.2f) %d(%d,%.2f)\n",node,level[node],sigma[node],adjacencyList[position],level[adjacencyList[position]],sigma[adjacencyList[position]]);
            // printf("g_delta[%d]: %f\n",g_csrE[position],g_delta[g_csrE[position]]);

        }
    }

}

__global__ void INITIAL_Qtruct(Q_struct* f1,int size){

    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size){
        f1[idx].nodeID = -1;
        f1[idx].traverse_S=0;
    }
  
}

__global__ void sum_BC_Result_MS(float* result,float* delta,int size,int* map_S,int mappingCount){
    extern __shared__ int shared_map_S[];
    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;

     // Load map_S into shared memory
    if (threadIdx.x < mappingCount) {
        shared_map_S[threadIdx.x] = map_S[threadIdx.x];
    }
    __syncthreads();

    if(idx < size){
        // #pragma unroll
        for(int i=0;i<mappingCount;i++){
            if(shared_map_S[i]==idx)
                continue;
            
            // atomicAdd(&result[idx], delta[idx*mappingCount+i]);
            result[idx] += delta[idx*mappingCount+i];
        }
        // printf("BC[%d]: %f\n",idx,result[idx]);
    }
        

}

__global__ void allBC_MS(int* g_csrV,int* g_csrE ,int* nextQueueSize,Q_struct* currentQueue,Q_struct* nextQueue,float* dist,int* sigma,int blocknum,int j,int size, int mappingcount){

    register const int bid = blockIdx.x + j * blocknum; // 0 + 0 * INT_MAX

    if(bid > size || currentQueue[bid].nodeID == -1) return; //大於currentQueueSize
    
    register const int node = currentQueue[bid].nodeID;
    register const uint64_t traverse_S = currentQueue[bid].traverse_S; //我改這裡
    register const int degree = g_csrV[node+1] - g_csrV[node];
    register const int threadOffset = (int)ceil(degree/(blockDim.x*1.0)); //需要看的鄰居，疊代次數
    register float     old;
    register int next;
    // printf("bid: %d,node: %d ,degree: %d, threadOffset: %d\n",bid,node,degree,threadOffset);
    for(int i=0;i<threadOffset;i++){
        register const int position = g_csrV[node] + threadIdx.x + i * blockDim.x; //該點node的鄰居位置
        if(position < g_csrV[node+1] ){
            register const int neighborNodeID = g_csrE[position];
            // printf("node: %d ,neighbor: %d threadOffset: %d\n",node,neighborNodeID,threadOffset);
            
            for (int multi_node = 0; multi_node < mappingcount; multi_node++) {
                // printf("multi_node: %d \n",multi_node);
                if (traverse_S & (1ULL << multi_node)) {
                    register const int position_v = mappingcount * node           + multi_node;
                    register const int position_n = mappingcount * neighborNodeID + multi_node;
                    // printf("node: %d ,neighbor: %d,pos_v: %d,pos_n: %d ,multi_node:%d\n",node,neighborNodeID,position_v,position_n,multi_node);
                    // 更新 dist_MULTI 和 sigma_MULTI
                    
                    // printf("ok1\n");
                    if (dist[position_n] > dist[position_v] + 1.0f) {
                        old = atomicMinFloat(&dist[position_n], (dist[position_v] + 1.0));
                        
                        if(old != dist[position_n]){
                            sigma[position_n] = 0;
                            // 检查 neighborNodeID 是否已在 nextQueue 中
                            // printf("node: %d ,neighbor: %d\n",node,neighborNodeID);
                            // printf("node: %d ,neighbor: %d\n",node,neighborNodeID);
                            next = atomicAdd(nextQueueSize,1);
                            nextQueue[next].nodeID = __ldg(&g_csrE[position]);
                            // nextQueue[next].nodeID     = neighborNodeID;
                            nextQueue[next].traverse_S = (1ULL << multi_node);
                            // bool found = false;
                            // for (int find = 0; find < next ; find++) {
                            //     // printf("enxtQueue[find].nodeID: %d\t neighborNodeID: %d\n",nextQueue[find].nodeID,neighborNodeID);
                            //     if (nextQueue[find].nodeID == neighborNodeID) {
                                    
                                    
                            //         // printf("node: %d ,neighbor: %d, nextQueueSize: %d\n",node,neighborNodeID,next);
                            //         nextQueue[find].traverse_S|=(1ULL << multi_node);
                            //         found = true;
                            //         atomicAdd(nextQueueSize,-1);
                            //         // break;
                            //     }
                            // }
                            
                            // if(found ==false){
                            //     nextQueue[next].nodeID = __ldg(&g_csrE[position]);
                            //     // printf("node: %d ,neighbor: %d, traverse_S: %d\n",node,neighborNodeID,(1ULL << multi_node));
                            //     // nextQueue[next].nodeID = neighborNodeID;
                            //     nextQueue[next].traverse_S = (1ULL << multi_node);
                            // }
                           
                        }
                        
                        // dist[position_n] = dist[position_v] + 1;
                        // sigma[position_n] = sigma[position_v];
                    }

                    if (dist[position_n] == dist[position_v] + 1.0f) {
                        atomicAdd(&sigma[position_n], sigma[position_v]);
                    }

                    // break;
                }
            }    
        }
    }

}

__global__ void allBC_MS_VnextQ(int* g_csrV,int* g_csrE ,Q_struct* currentQueue,Q_struct* nextQueue,float* dist,int* sigma,int blocknum,int j,int size, int mappingcount){

    register const int bid = blockIdx.x + j * blocknum; // 0 + 0 * INT_MAX

    if(bid > size || currentQueue[bid].nodeID == -1) return; //大於currentQueueSize
    
    register const int node = currentQueue[bid].nodeID;
    register const uint64_t traverse_S = currentQueue[bid].traverse_S; //我改這裡
    register const int degree = g_csrV[node+1] - g_csrV[node];
    register const int threadOffset = (int)ceil(degree/(blockDim.x*1.0)); //需要看的鄰居，疊代次數
    register float     old;
    // printf("bid: %d,node: %d ,degree: %d, threadOffset: %d\n",bid,node,degree,threadOffset);
    for(int i=0;i<threadOffset;i++){
        register const int position = g_csrV[node] + threadIdx.x + i * blockDim.x; //該點node的鄰居位置
        if(position < g_csrV[node+1] ){
            register const int neighborNodeID = g_csrE[position];
            // printf("node: %d ,neighbor: %d threadOffset: %d\n",node,neighborNodeID,threadOffset);
            
            for (int multi_node = 0; multi_node < mappingcount; multi_node++) {
                // printf("multi_node: %d \n",multi_node);
                if (traverse_S & (1ULL << multi_node)) {
                    register const int position_v = mappingcount * node           + multi_node;
                    register const int position_n = mappingcount * neighborNodeID + multi_node;
                    // printf("node: %d ,neighbor: %d,pos_v: %d,pos_n: %d ,multi_node:%d\n",node,neighborNodeID,position_v,position_n,multi_node);
                    // 更新 dist_MULTI 和 sigma_MULTI
                    
                    // printf("ok1\n");
                    if (dist[position_n] > dist[position_v] + 1.0f) {
                        old = atomicMinFloat(&dist[position_n], (dist[position_v] + 1.0));
                        
                        if(old != dist[position_n]){
                            sigma[position_n] = 0;
                            // 检查 neighborNodeID 是否已在 nextQueue 中
                            // printf("node: %d ,neighbor: %d\n",node,neighborNodeID);
                            // printf("node: %d ,neighbor: %d\n",node,neighborNodeID);
                            nextQueue[g_csrE[position]].nodeID = __ldg(&g_csrE[position]);
                            // nextQueue[g_csrE[position]].traverse_S |= (1ULL << multi_node);
                            atomicOr64(&nextQueue[g_csrE[position]].traverse_S,(1ULL << multi_node));
                            
                            
                            // bool found = false;
                            // for (int find = 0; find < next ; find++) {
                            //     // printf("enxtQueue[find].nodeID: %d\t neighborNodeID: %d\n",nextQueue[find].nodeID,neighborNodeID);
                            //     if (nextQueue[find].nodeID == neighborNodeID) {
                                    
                                    
                            //         // printf("node: %d ,neighbor: %d, nextQueueSize: %d\n",node,neighborNodeID,next);
                            //         nextQueue[find].traverse_S|=(1ULL << multi_node);
                            //         found = true;
                            //         atomicAdd(nextQueueSize,-1);
                            //         // break;
                            //     }
                            // }
                            
                            // if(found ==false){
                            //     nextQueue[next].nodeID = __ldg(&g_csrE[position]);
                            //     // printf("node: %d ,neighbor: %d, traverse_S: %d\n",node,neighborNodeID,(1ULL << multi_node));
                            //     // nextQueue[next].nodeID = neighborNodeID;
                            //     nextQueue[next].traverse_S = (1ULL << multi_node);
                            // }
                           
                        }
                        
                        // dist[position_n] = dist[position_v] + 1;
                        // sigma[position_n] = sigma[position_v];
                    }

                    if (dist[position_n] == dist[position_v] + 1.0f) {
                        atomicAdd(&sigma[position_n], sigma[position_v]);
                    }

                    // break;
                }
            }    
        }
    }

}


__global__ void rearrange_queue_MS(Q_struct* nextQueue,Q_struct* nextQueue_temp,int* nextQueueSize, const int V) { 
    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;    
    register int next;
    if(idx < V) {
        if(nextQueue_temp[idx].nodeID!=-1){
            next = atomicAdd(nextQueueSize,1);
            nextQueue[next]=nextQueue_temp[idx];
        }
    }
}


//這個版本為1，相同source的分開不同block的，速度變慢
void brandes_MS_par( CSR& csr, int max_multi, float* BC) {
    // Start timing
    // multi_time_start = seconds();

    int V = csr.csrVSize;
    int multi_size = V * max_multi;
    //CPU variable
    int    currentQueueSize;
    int*   stackOffset = (int*)calloc(multi_size,sizeof(int));
    int*   map_S = (int*)malloc(sizeof(int) * max_multi); // Multiple sources
    bool*  nodeDone = (bool*)calloc(V, sizeof(bool));
    //CPU print 專用
    int*   sigma    = (int*)malloc(sizeof(int) * multi_size);
    float* dist     = (float*)malloc(sizeof(float) * multi_size);
    Q_struct*  f1   = (Q_struct*)malloc(sizeof(Q_struct) * V);
    Q_struct*  queue   = (Q_struct*)malloc(sizeof(Q_struct) * V);
    float*  delta   = (float*)malloc(sizeof(float) * multi_size);
    //GPU MALLOC　variable
    Q_struct*   g_stack;       
    int*   g_sigma;     
    float* g_dist;
    int*   g_level;     
    float* g_delta; 
    int*   g_S_size;
    Q_struct*   g_f1;
    Q_struct*   g_f2;
    Q_struct*   nextQueue_temp;
    int*   g_nextQueueSize; //用來回傳給CPU判別currentQueueSize，是否繼續traverse
    int*   g_csrE;
    int*   g_csrV;
    int*   g_map_S;   //用來記錄Source對應的node array位置: 0 1 2 -> node 22 15 6
    float* g_BC;


    // printf("start malloc\n");
    cudaMalloc((void **)&g_stack,multi_size * sizeof(Q_struct)); //用CPU的stack offset存每一層的位置 因為node可能在不同層重複出現，所以要開multi_size的大小
    cudaMalloc((void **)&nextQueue_temp,V * sizeof(Q_struct));
    cudaMalloc((void **)&g_sigma,multi_size * sizeof(int));
    cudaMalloc((void **)&g_dist,multi_size * sizeof(float));
    cudaMalloc((void **)&g_level,V * sizeof(int));
    cudaMalloc((void **)&g_delta,multi_size * sizeof(float));
    cudaMalloc((void **)&g_S_size,V* sizeof(int));
    
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte,&total_byte);
    cudaMalloc((void **)&g_f1, multi_size * sizeof(Q_struct)); //free_byte * 0.3
    cudaMalloc((void **)&g_f2, multi_size * sizeof(Q_struct)); //free_byte * 0.3
    cudaMalloc((void **)&g_nextQueueSize,sizeof(int));
    cudaMalloc((void **)&g_csrV, V * sizeof(int));
    cudaMalloc((void **)&g_csrE, csr.csrESize * sizeof(int));
    cudaMalloc((void **)&g_BC, V * sizeof(float));
    cudaMalloc((void **)&g_map_S, max_multi * sizeof(int));
    cudaMemcpy(g_csrV,csr.csrV ,  V * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(g_csrE,csr.csrE ,  csr.csrESize * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(g_BC, 0.0f, V * sizeof(float));
    // printf("end malloc\n");
    // std::cout << "Total GPU memory: " << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << "Free GPU memory: " << free_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    int threadnum = 32;


    //origin

    for (int sourceID = csr.startNodeID; sourceID <= csr.endNodeID; ++sourceID) {
        if (nodeDone[sourceID]) continue;

        // multi_time1 = seconds();

        nodeDone[sourceID] = true;
        int mappingCount = 0;
        map_S[mappingCount++] = sourceID;

        // Find other sources
        for (int neighborIndex = csr.csrV[sourceID]; neighborIndex < csr.csrV[sourceID + 1] && mappingCount < max_multi; neighborIndex++) {
            int neighborNodeID = csr.csrE[neighborIndex];
            if (!nodeDone[neighborNodeID]) {
                map_S[mappingCount++] = neighborNodeID;
                nodeDone[neighborNodeID] = true;
            }
        }
    
        //亂挑multi-source的程式
        // for (int neighborIndex = csr.startNodeID; neighborIndex <= csr.endNodeID && mappingCount < max_multi; neighborIndex++) {
        //     int neighborNodeID = csr.csrE[neighborIndex];
        //     if (!nodeDone[neighborNodeID]) {
        //         map_S[mappingCount++] = neighborNodeID;
        //         nodeDone[neighborNodeID] = true;
        //     }
        // }


        // Initialize dist_MULTI, sigma_MULTI, delta_MULTI
        //初始g_f1 queue
        cudaMemcpy(g_map_S,map_S, mappingCount * sizeof(int),cudaMemcpyHostToDevice);
        resetBC_value_MS<<<ceil(multi_size/128.0),min(multi_size,128)>>>(g_dist,g_f1,g_f2,g_sigma,g_delta,g_stack,sourceID,multi_size);
        cudaMemset(g_nextQueueSize,0,sizeof(int));
        currentQueueSize = mappingCount;
        memset(stackOffset, 0, sizeof(int) * multi_size);
        INITIAL_value_MS<<<ceil(mappingCount/128.0),min(mappingCount,128)>>>(g_dist,g_f1,g_sigma,g_map_S,mappingCount);
        cudaDeviceSynchronize();
        #pragma  region print
        //檢查GPU資料
        // cudaMemcpy(sigma,g_sigma, multi_size * sizeof(int),cudaMemcpyHostToHost);
        // cudaMemcpy(dist, g_dist,   multi_size * sizeof(float),cudaMemcpyHostToHost);
        // cudaMemcpy(f1, g_f1,   mappingCount * sizeof(Q_struct),cudaMemcpyDeviceToHost);

        // printf("sigma: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%d,",sigma[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");

        // printf("dist: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%.0f,",dist[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");

        // printf("--------------------multi Source--------------------");
        // for(int i=0;i<max_multi;i++){
        //     printf("%d ",map_S[i]);
        // }
        // printf("\n");

        // printf("f1: ");
        // for(int i=0;i<multi_size;i++){
        //     printf("[%d, %d]> ",f1[i].nodeID,f1[i].traverse_S);
        // }
        // printf("\n");
        #pragma  endregion

        // int f1_indicator = 0;
        // int f2_indicator = 0;
        // int s_indicator = 0;
       

        int level=0;
        while (currentQueueSize > 0) { //currentQueueSize > 0
            // std::cout<<"currentQueueSize: "<<currentQueueSize<<std::endl;
            Q_struct* g_currentQueue;
            Q_struct* g_nextQueue;
            if(level% 2 == 0){
                g_currentQueue = g_f1;
                g_nextQueue = g_f2;
            }
            else{
                g_currentQueue = g_f2;
                g_nextQueue = g_f1;
            }
            
            stackOffset[level+1] = currentQueueSize + stackOffset[level];
            int blocknum = (currentQueueSize < INT_MAX) ? currentQueueSize : INT_MAX;
            //平行跑BFS
            // printf("currentQueueSize: %d\n",currentQueueSize);
            for(int i=0;i<(int)ceil(currentQueueSize/(float)INT_MAX);i++){
                allBC_MS<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_nextQueueSize,g_currentQueue,g_nextQueue,g_dist,g_sigma,INT_MAX,i,currentQueueSize,mappingCount);
                // allBC_MS_VnextQ<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_currentQueue,g_nextQueue,g_dist,g_sigma,INT_MAX,i,currentQueueSize,mappingCount);
                cudaDeviceSynchronize();
                // CHECK(cudaMemcpy(&nextQueueSize_temp,g_nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
                // int shared_mem_size = (*g_nextQueueSize) * sizeof(Q_struct);
                // rearrange_queue_MS<<<1,1>>>(nextQueue_temp,g_nextQueueSize,g_nextQueue);
                // cudaDeviceSynchronize();
            }
                    

            // Swap currentQueue and nextQueue
            // CHECK(cudaMemcpy(&currentQueueSize,g_nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
            // CHECK(cudaMemcpy(&g_stack[stackOffset[level+1]],g_nextQueue,currentQueueSize*sizeof(Q_struct),cudaMemcpyDeviceToDevice));
            // CHECK(cudaMemset(g_nextQueueSize,0,sizeof(int)));
            cudaMemcpy(&currentQueueSize,g_nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(&g_stack[stackOffset[level+1]],g_nextQueue,currentQueueSize*sizeof(Q_struct),cudaMemcpyDeviceToDevice);
            cudaMemset(g_nextQueueSize,0,sizeof(int));
            level++;


            #pragma  region print
            // CHECK(cudaMemcpy(&queue[0],g_nextQueue, currentQueueSize*sizeof(Q_struct),cudaMemcpyDeviceToHost));
            // printf("f1: ");
            // for(int i=0;i<currentQueueSize;i++){
            //     printf("[%d, %lu]> ",queue[i].nodeID,queue[i].traverse_S);
            // }
            // printf("\n");
            #pragma  endregion


        }

        #pragma  region print
        //檢查GPU資料
        // cudaMemcpy(sigma,g_sigma, multi_size * sizeof(int),cudaMemcpyDeviceToHost);
        // cudaMemcpy(dist, g_dist,   multi_size * sizeof(float),cudaMemcpyDeviceToHost);
        // cudaMemcpy(f1, g_f1,   mappingCount * sizeof(Q_struct),cudaMemcpyDeviceToHost);

        // printf("sigma: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%d,",sigma[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");

        // printf("dist: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%.0f,",dist[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");


        // printf("f1: ");
        // for(int i=0;i<multi_size;i++){
        //     printf("[%d, %d]> ",f1[i].nodeID,f1[i].traverse_S);
        // }
        // printf("\n");
        #pragma  endregion



        // multi_time2 = seconds();
        // multi_forward_Time += (multi_time2 - multi_time1);
        // multi_time1 = seconds();

        // Back-propagation
        //  std::cout << "--------------backward--------------"<< std::endl;
        for (int d = level - 1; d > 0; d--) {
            int degree =(stackOffset[d+1] - stackOffset[d]);
            int blocknum = (degree < INT_MAX) ? degree : INT_MAX;
            // std::cout << "backward level(" << d << "):\t" << stackOffset[d+1] - stackOffset[d] << std::endl;
            for(int i=0;i<(int)ceil(degree/(float)INT_MAX);i++){
                deltaCalculation_MS<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_delta,g_sigma,g_stack,g_dist,INT_MAX,i,stackOffset[d],degree,mappingCount);
                CHECK(cudaDeviceSynchronize());
            }
                
            // cudaDeviceSynchronize();
            #pragma  region print
            // CHECK(cudaMemcpy(&delta[0], g_delta,   multi_size * sizeof(float),cudaMemcpyDeviceToHost));
            // printf("delta: ");
            // for(int i=0;i<V;i++){
            //     printf("[");
            //     for(int j=0;j<mappingCount;j++){
            //         printf("%.3f,",delta[mappingCount*i+j]);
            //     }
            //     printf("] ");
            // }
            // printf("\n");
            #pragma  endregion
        }
        int shared_mem_size = (mappingCount) * sizeof(int);
        sum_BC_Result_MS<<<ceil(V/128.0),min(V,128),shared_mem_size>>>(g_BC,g_delta,V,g_map_S,mappingCount);
        CHECK(cudaDeviceSynchronize());

        

        // multi_time2 = seconds();
        // multi_backward_Time += (multi_time2 - multi_time1);
    }
    CHECK(cudaMemcpy(&BC[0],g_BC, V*sizeof(float),cudaMemcpyDeviceToHost));
    // multi_time_end = seconds();
    // multi_total_time = (multi_time_end - multi_time_start);

   
}

//這個版本為1，相同source的擠在相同block的，速度快
void brandes_MS_par_VnextQ( CSR& csr, int max_multi, float* BC) {
    // Start timing
    // multi_time_start = seconds();

    int V = csr.csrVSize;
    int multi_size = V * max_multi;
    //CPU variable
    int    currentQueueSize;
    int*   stackOffset = (int*)calloc(multi_size,sizeof(int));
    int*   map_S = (int*)malloc(sizeof(int) * max_multi); // Multiple sources
    bool*  nodeDone = (bool*)calloc(V, sizeof(bool));
    //CPU print 專用
    int*   sigma    = (int*)malloc(sizeof(int) * multi_size);
    float* dist     = (float*)malloc(sizeof(float) * multi_size);
    Q_struct*  f1   = (Q_struct*)malloc(sizeof(Q_struct) * V);
    Q_struct*  queue   = (Q_struct*)malloc(sizeof(Q_struct) * multi_size);
    float*  delta   = (float*)malloc(sizeof(float) * multi_size);
    //GPU MALLOC　variable
    Q_struct*   g_stack;       
    int*   g_sigma;     
    float* g_dist;
    int*   g_level;     
    float* g_delta; 
    int*   g_S_size;
    Q_struct*   g_f1;
    Q_struct*   g_f2;
    Q_struct*   g_nextQueue_temp;
    int*   g_nextQueueSize; //用來回傳給CPU判別currentQueueSize，是否繼續traverse
    int*   g_csrE;
    int*   g_csrV;
    int*   g_map_S;   //用來記錄Source對應的node array位置: 0 1 2 -> node 22 15 6
    float* g_BC;


    // printf("start malloc\n");
    cudaMalloc((void **)&g_stack,multi_size * sizeof(Q_struct)); //用CPU的stack offset存每一層的位置 因為node可能在不同層重複出現，所以要開multi_size的大小
    cudaMalloc((void **)&g_sigma,multi_size * sizeof(int));
    cudaMalloc((void **)&g_dist,multi_size * sizeof(float));
    cudaMalloc((void **)&g_level,V * sizeof(int));
    cudaMalloc((void **)&g_delta,multi_size * sizeof(float));
    cudaMalloc((void **)&g_S_size,V* sizeof(int));
    
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte,&total_byte);
    cudaMalloc((void **)&g_f1, multi_size * sizeof(Q_struct)); //free_byte * 0.3
    cudaMalloc((void **)&g_f2, multi_size * sizeof(Q_struct)); //free_byte * 0.3
    cudaMalloc((void **)&g_nextQueue_temp,V * sizeof(Q_struct));
    cudaMalloc((void **)&g_nextQueueSize,sizeof(int));
    cudaMalloc((void **)&g_csrV, V * sizeof(int));
    cudaMalloc((void **)&g_csrE, csr.csrESize * sizeof(int));
    cudaMalloc((void **)&g_BC, V * sizeof(float));
    cudaMalloc((void **)&g_map_S, max_multi * sizeof(int));
    cudaMemcpy(g_csrV,csr.csrV ,  V * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(g_csrE,csr.csrE ,  csr.csrESize * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(g_BC, 0.0f, V * sizeof(float));
    // printf("end malloc\n");
    // std::cout << "Total GPU memory: " << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << "Free GPU memory: " << free_byte / (1024.0 * 1024.0) << " MB" << std::endl;
    int threadnum = 32;

    //order ID by degree
    csr.orderedCsrV  = (int*)calloc(sizeof(int), (csr.csrVSize) *2);
    for(int i=csr.startNodeID;i<=csr.endNodeID;i++){
            csr.orderedCsrV[i]=i;
    }
    quicksort_nodeID_with_degree(csr.orderedCsrV, csr.csrNodesDegree, csr.startNodeID, csr.endNodeID);


    //origin
    for (int sourceIndex = csr.startNodeID; sourceIndex <= csr.endNodeID; ++sourceIndex) {
        int sourceID=csr.orderedCsrV[sourceIndex];
        if (nodeDone[sourceID]) continue;

        // multi_time1 = seconds();

        nodeDone[sourceID] = true;
        int mappingCount = 0;
        map_S[mappingCount++] = sourceID;

        // Find other sources
        // for (int neighborIndex = csr.csrV[sourceID]; neighborIndex < csr.csrV[sourceID + 1] && mappingCount < max_multi; neighborIndex++) {
        //     int neighborNodeID = csr.csrE[neighborIndex];
        //     if (!nodeDone[neighborNodeID]) {
        //         map_S[mappingCount++] = neighborNodeID;
        //         nodeDone[neighborNodeID] = true;
        //     }
        // }
        for (int neighborIndex = csr.startNodeID; neighborIndex <= csr.endNodeID && mappingCount < max_multi; neighborIndex++) {
            int neighborNodeID = csr.orderedCsrV[neighborIndex];
            if (!nodeDone[neighborNodeID]) {
                map_S[mappingCount++] = neighborNodeID;
                nodeDone[neighborNodeID] = true;
            }
        }


        // Initialize dist_MULTI, sigma_MULTI, delta_MULTI
        //初始g_f1 queue
        cudaMemcpy(g_map_S,map_S, mappingCount * sizeof(int),cudaMemcpyHostToDevice);
        resetBC_value_MS<<<ceil(multi_size/64.0),min(multi_size,64)>>>(g_dist,g_f1,g_f2,g_sigma,g_delta,g_stack,sourceID,multi_size);
        cudaDeviceSynchronize();
        cudaMemset(g_nextQueueSize,0,sizeof(int));
        currentQueueSize = mappingCount;
        memset(stackOffset, 0, sizeof(int) * multi_size);
        INITIAL_value_MS<<<ceil(mappingCount/64.0),min(mappingCount,64)>>>(g_dist,g_f1,g_sigma,g_map_S,mappingCount);
        cudaDeviceSynchronize();
        #pragma  region print
        //檢查GPU資料
        // cudaMemcpy(sigma,g_sigma, multi_size * sizeof(int),cudaMemcpyHostToHost);
        // cudaMemcpy(dist, g_dist,   multi_size * sizeof(float),cudaMemcpyHostToHost);
        // cudaMemcpy(f1, g_f1,   mappingCount * sizeof(Q_struct),cudaMemcpyDeviceToHost);

        // printf("sigma: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%d,",sigma[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");

        // printf("dist: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%.0f,",dist[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");

        // printf("--------------------multi Source-------------------- ");
        // for(int i=0;i<mappingCount;i++){
        //     printf("%d ",map_S[i]);
        // }
        // printf("\n");

        // printf("f1: ");
        // for(int i=0;i<multi_size;i++){
        //     printf("[%d, %d]> ",f1[i].nodeID,f1[i].traverse_S);
        // }
        // printf("\n");
        #pragma  endregion

        // int f1_indicator = 0;
        // int f2_indicator = 0;
        // int s_indicator = 0;
       

        int level=0;
        while (currentQueueSize > 0) { //currentQueueSize > 0
            // std::cout<<"currentQueueSize: "<<currentQueueSize<<std::endl;
            Q_struct* g_currentQueue;
            Q_struct* g_nextQueue;
            INITIAL_Qtruct<<<ceil(V/64.0),min(V,64)>>>(g_nextQueue_temp,V);
            cudaDeviceSynchronize();
            
            int index = level & 1;  // 等價於 level % 2
            g_currentQueue = (index == 0) ? g_f1 : g_f2;
            g_nextQueue    = (index == 0) ? g_f2 : g_f1;
            
            stackOffset[level+1] = currentQueueSize + stackOffset[level];
            int blocknum = (currentQueueSize < INT_MAX) ? currentQueueSize : INT_MAX;
            //平行跑BFS
            // printf("currentQueueSize: %d\n",currentQueueSize);
            for(int i=0;i<(int)ceil(currentQueueSize/(float)INT_MAX);i++){
                // allBC_MS<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_nextQueueSize,g_currentQueue,g_nextQueue,g_dist,g_sigma,INT_MAX,i,currentQueueSize,mappingCount);
                allBC_MS_VnextQ<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_currentQueue,g_nextQueue_temp,g_dist,g_sigma,INT_MAX,i,currentQueueSize,mappingCount);
                cudaDeviceSynchronize();
                rearrange_queue_MS<<<ceil(V/64.0),min(V,64)>>>(g_nextQueue,g_nextQueue_temp,g_nextQueueSize,V);
                cudaDeviceSynchronize();
            }
                    

            // Swap currentQueue and nextQueue
            // CHECK(cudaMemcpy(&currentQueueSize,g_nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost));
            // CHECK(cudaMemcpy(&g_stack[stackOffset[level+1]],g_nextQueue,currentQueueSize*sizeof(Q_struct),cudaMemcpyDeviceToDevice));
            // CHECK(cudaMemset(g_nextQueueSize,0,sizeof(int)));
            cudaMemcpy(&currentQueueSize,g_nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(&g_stack[stackOffset[level+1]],g_nextQueue,currentQueueSize*sizeof(Q_struct),cudaMemcpyDeviceToDevice);
            cudaMemset(g_nextQueueSize,0,sizeof(int));
            level++;


            #pragma  region print
            // CHECK(cudaMemcpy(&queue[0],g_nextQueue, currentQueueSize*sizeof(Q_struct),cudaMemcpyDeviceToHost));
            // printf("real_f1: ");
            // for(int i=0;i<currentQueueSize;i++){
            //     printf("[%d, %lu]> ",queue[i].nodeID,queue[i].traverse_S);
            // }
            // printf("\n");
            #pragma  endregion


        }

        #pragma  region print
        //檢查GPU資料
        // cudaMemcpy(sigma,g_sigma, multi_size * sizeof(int),cudaMemcpyDeviceToHost);
        // cudaMemcpy(dist, g_dist,   multi_size * sizeof(float),cudaMemcpyDeviceToHost);
        // cudaMemcpy(f1, g_f1,   mappingCount * sizeof(Q_struct),cudaMemcpyDeviceToHost);

        // printf("sigma: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%d,",sigma[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");

        // printf("dist: ");
        // for(int i=0;i<V;i++){
        //     printf("[");
        //     for(int j=0;j<mappingCount;j++){
        //         printf("%.0f,",dist[mappingCount*i+j]);
        //     }
        //     printf("] ");
        // }
        // printf("\n");


        // printf("f1: ");
        // for(int i=0;i<multi_size;i++){
        //     printf("[%d, %d]> ",f1[i].nodeID,f1[i].traverse_S);
        // }
        // printf("\n");
        #pragma  endregion



        // multi_time2 = seconds();
        // multi_forward_Time += (multi_time2 - multi_time1);
        // multi_time1 = seconds();

        // Back-propagation
        //  std::cout << "--------------backward--------------"<< std::endl;
        for (int d = level - 1; d > 0; d--) {
            int degree =(stackOffset[d+1] - stackOffset[d]);
            int blocknum = (degree < INT_MAX) ? degree : INT_MAX;
            // std::cout << "backward level(" << d << "):\t" << stackOffset[d+1] - stackOffset[d] << std::endl;
            for(int i=0;i<(int)ceil(degree/(float)INT_MAX);i++){
                deltaCalculation_MS<<<blocknum,threadnum>>>(g_csrV,g_csrE,g_delta,g_sigma,g_stack,g_dist,INT_MAX,i,stackOffset[d],degree,mappingCount);
                CHECK(cudaDeviceSynchronize());
            }
                
            // cudaDeviceSynchronize();
            #pragma  region print
            // CHECK(cudaMemcpy(&delta[0], g_delta,   multi_size * sizeof(float),cudaMemcpyDeviceToHost));
            // printf("delta: ");
            // for(int i=0;i<V;i++){
            //     printf("[");
            //     for(int j=0;j<mappingCount;j++){
            //         printf("%.3f,",delta[mappingCount*i+j]);
            //     }
            //     printf("] ");
            // }
            // printf("\n");
            #pragma  endregion
        }
        int shared_mem_size = (mappingCount) * sizeof(int);
        sum_BC_Result_MS<<<ceil(V/128.0),min(V,128),shared_mem_size>>>(g_BC,g_delta,V,g_map_S,mappingCount);
        CHECK(cudaDeviceSynchronize());

        

        // multi_time2 = seconds();
        // multi_backward_Time += (multi_time2 - multi_time1);
    }
    CHECK(cudaMemcpy(&BC[0],g_BC, V*sizeof(float),cudaMemcpyDeviceToHost));



    // multi_time_end = seconds();
    // multi_total_time = (multi_time_end - multi_time_start);
    // 釋放所有 GPU 資源
    cudaFree(g_stack);
    cudaFree(g_sigma);
    cudaFree(g_dist);
    cudaFree(g_level);
    cudaFree(g_delta);
    cudaFree(g_S_size);
    cudaFree(g_f1);
    cudaFree(g_f2);
    cudaFree(g_nextQueue_temp);
    cudaFree(g_nextQueueSize);
    cudaFree(g_csrE);
    cudaFree(g_csrV);
    cudaFree(g_map_S);
    cudaFree(g_BC);
   
}


