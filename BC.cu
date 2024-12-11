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
using namespace std;
#include "headers.h"
#define INFINITE 1000000000
// #define DEBUGx


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

#pragma endregion //globalVar


inline void resetQueue(struct qQueue* _Q){
    _Q->front   = 0;
    _Q->rear    = -1;
    //Q->size如果不變，就不須memcpy
}

void check_ans(std::vector<float> ans, std::vector<float> my_ans);
void brandes_ORIGIN_for_Seq( CSR& csr, int V, vector<float> &BC);
void Seq_multi_source_brandes_ordered( CSR& csr, int max_multi, vector<float> &BC);
void Seq_multi_source_brandes(CSR& csr, int max_multi, vector<float> &BC);
void brandes_SS_par( CSR& csr, int V, float *BC);
void brandes_MS_par( CSR& csr, int max_multi, float* BC);
void brandes_MS_par_VnextQ( CSR& csr, int max_multi, float* BC);
void Seq_MS_brandes_me_D1_AP(CSR* csr, int max_multi, vector<float> &BC);
void brandes_MS_Me_AP_D1( CSR& csr, int max_multi, float* BC);


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


    time1 = seconds();
    // brandes_ORIGIN_for_Seq(*csr,csr->csrVSize,ans);
    // // brandes_SS_par(*csr,csr->csrVSize,ans_para);
    // brandes_MS_par(*csr , max_multi , ans_para);
    // // brandes_MS_par(*csr , max_multi , ans_para);
    time2 = seconds();
    printf("done brandes_MS_par\n");


    multi_time1 = seconds();
    // Seq_multi_source_brandes_ordered( *csr , max_multi , my_BC );
    // brandes_ORIGIN_for_Seq(*csr,csr->csrVSize,my_BC);
    // brandes_SS_par(*csr,csr->csrVSize,ans_para);
    // brandes_MS_par_VnextQ(*csr , max_multi , ans_para2); 
    // brandes_MS_Me_para(*csr , max_multi , ans_para2); 
    Seq_MS_brandes_me_D1_AP(csr , max_multi , ans_para_vec); 
    multi_time2 = seconds();
    printf("done brandes_SS_par\n");

    // computeBC_shareBased(csr,my_BC);
    // Seq_multi_source_brandes( *csr , max_multi , my_BC );


    //檢查答案
    // for(int i=0;i<csr->csrVSize;i++){
    //     ans_para_vec[i]=ans_para[i];
    //     ans_para_vec2[i]=ans_para2[i];
    // }
    // check_ans(ans,ans_para_vec2);
    

    #ifdef DEBUG
        for(auto i=0;i<csr->csrVSize;i++){
            printf("BC[%d]: %f\n",i,ans[i]);
        }
    #endif
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
    
    
    printf("[Execution Time] total_time        = %.6f secs\n", time2-time1);
    printf("[Execution Time] OMP_total_time    = %.6f secs\n", multi_time2-multi_time1);
    printf("[Execution Time] speedup ratio     = %.6f secs\n", (time2-time1)/(multi_time2-multi_time1));
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
        std::cout << "[CORRECT] my_ans matches ans within 1% tolerance!" << std::endl;
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

//循序_brandes原版
void brandes_ORIGIN_for_Seq( CSR& csr, int V, vector<float> &BC) {
    // time_start = seconds();

  

    // Allocate memory for sigma, dist, delta, and the stack S
    int**  S = (int**)malloc(V * sizeof(int*));      // S is a 2D array (stack)
    int*   sigma = (int*)malloc(V * sizeof(int));     // sigma is a 1D array
    int*   dist = (int*)malloc(V * sizeof(int));      // dist is a 1D array
    float* delta = (float*)malloc(V * sizeof(float)); // delta is a 1D array
    int*   S_size = (int*)malloc(V * sizeof(int));    // S_size records the size of each level
    int*   f1 = (int*)malloc((V ) * sizeof(int));
    int*   f2 = (int*)malloc((V ) * sizeof(int));
    int    f1_indicator;
    int    f2_indicator;

    for (int i = 0; i < V; i++) {
        S[i] = (int*)malloc(V *sizeof(int));       // Each level's stack size V, adjust as needed
        S_size[i] = 0;                              // Initialize the size of each level
    }

    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        // Initialize variables for each source node
        // time1 = seconds();
        for (int i = 0; i < V; i++) {
            free(S[i]);
            S[i] = (int*)malloc(V*sizeof(int));   // Each level's stack size V, adjust as needed
            sigma[i] = 0;
            dist[i] = -1;
            delta[i] = 0.0;
            S_size[i] = 0; // Reset size of each level
        }
    
        sigma[s] = 1;
        dist[s] = 0;
        f1_indicator = 0;
        f2_indicator = 0;
        // Re-initialize current_queue
        f1[f1_indicator++] = s;
        

        int level =0;
        // BFS forward phase: frontier-based BFS with extra mallocs
        while (f1_indicator>0) { //!qIsEmpty(current_queue)
            // printf("level: %d\n",level);
            
            // Allocate new memory for next_queue in each iteration
            int* currentQueue;
            int* nextQueue;
            if(level% 2 == 0){
                currentQueue = f1;
                nextQueue = f2;
            }
            else{
                currentQueue = f2;
                nextQueue = f1;
            }
            
            for(int v=0; v<f1_indicator; v++) {
                int u = currentQueue[v];
                S[level][S_size[level]++] = u;  // Put node u into its level
                
                // Traverse the adjacent nodes in CSR format
                for (int i = csr.csrV[u]; i < csr.csrV[u + 1]; ++i) {
                    int w = csr.csrE[i];
                    
                    // If w has not been visited, update distance and add to next_queue
                    if (dist[w] < 0) {
                        dist[w] = dist[u] + 1;
                        nextQueue[f2_indicator++] = w;
                        
                    }

                    // When a shortest path is found
                    if (dist[w] == dist[u] + 1) {
                        sigma[w] += sigma[u];
                    }
                }
            }
            
            // Free current_queue and set it to next_queue for the next iteration
            f1_indicator = f2_indicator;
            f2_indicator = 0;
            level++;
            
            // printf("level: %d\n",level);
            // qShowAllElement(current_queue);
            
        }

        // time2 = seconds();
        // forward_Time += (time2 - time1);
        // time1 = seconds();

        // Backward phase to compute BC values
        for (int d = level - 1; d >= 0; --d) {  // Start from the furthest level
            
            for (int i = 0; i < S_size[d]; ++i) {
                int w = S[d][i];
                
                for (int j = csr.csrV[w]; j < csr.csrV[w + 1]; ++j) {
                    int v = csr.csrE[j];
                    if (dist[v] == dist[w] - 1) {
                        delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
                    }
                }
                if (w != s) {
                    BC[w] += delta[w];
                }
            }
            
        }
        // time2 = seconds();
        // backward_Time += (time2 - time1);
    }

    // time_end = seconds();
    // total_time = (time_end - time_start);
    // Free memory for S and its levels
    for (int i = 0; i < V; i++) {
        free(S[i]);
    }
    free(S);
    free(sigma);
    free(dist);
    free(delta);
    free(S_size);
}

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


