// #ifndef COMMON
// #define COMMON
#include <stdio.h>
#include <cmath> // 引入 floor 函數
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <iomanip>  // 需要包含這個頭文件以使用 setprecision
#include <cuda_runtime.h>
#include "device_atomic_functions.h"
// #endif

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


void brandes(CSR& csr, int V, float* BC);
void bellman_D2(CSR& csr, int V, float* BC);
void bellman_all_Degree(CSR& csr, int V, float* BC_ans);

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


    multi_time1 = seconds();
    // bellman_D2(*csr,csr->csrVSize,ans_para);
    bellman_all_Degree(*csr,csr->csrVSize,ans_para);
    multi_time2 = seconds();
    printf("done 2\n");

    

    
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
//                   循序_各種centrality
//************************************************ */

void brandes(CSR& csr, int V, float* BC) {
    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);                // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> predecessors(V);   // Predecessor list

    long long total_predecessor_count = 0; // To accumulate total predecessors

    double time_phase1=0.0;
    double time_phase2=0.0;
    double start_time=0.0;
    double end_time=0.0;

    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        start_time=seconds();
        // Initialize arrays for each source node
        sigma.assign(V, 0);   // Reset sigma to size V with all values 0
        dist.assign(V, -1);   // Reset dist to size V with all values -1
        delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
        // fill(sigma.begin(), sigma.end(), 0);
        // fill(dist.begin(), dist.end(), -1);
        // fill(delta.begin(), delta.end(), 0.0);
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

void stress_centrality(CSR& csr, int V, float* stress) {
    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);                // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> predecessors(V);   // Predecessor list

    double time_phase1 = 0.0;
    double time_phase2 = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;

    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        start_time = seconds();

        // Initialize arrays for each source node
        sigma.assign(V, 0);   // Reset sigma to size V with all values 0
        dist.assign(V, -1);   // Reset dist to size V with all values -1
        delta.assign(V, 0.0); // Reset delta to size V with all values 0.0
        S.assign(V, vector<int>());  // Reset S with empty vectors
        predecessors.assign(V, vector<int>());  // Reset predecessors with empty vectors

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

        end_time = seconds();
        time_phase1 += end_time - start_time;
        start_time = seconds();

        // Backward phase: Stress Centrality Computation
        for (int d = level - 1; d >= 0; --d) {
            for (int w : S[d]) {
                for (int v : predecessors[w]) {
                    delta[v] += (1.0 + delta[w]);
                }
                if (w != s) {
                    stress[w] += (sigma[w] + delta[w]);
                }
            }
        }

        end_time = seconds();
        time_phase2 += end_time - start_time;
    }

    printf("phase1 time: %0.6f\n", time_phase1);
    printf("phase2 time: %0.6f\n", time_phase2);
}


//************************************************ */
//             循序_Bellman criterion測試
//************************************************ */
void bellman_D2(CSR& csr, int V, float* BC_ans) {
    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);                // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    // vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> predecessors(V);   // Predecessor list

    vector<vector<int>> Source_path(V,vector<int>(V,0));    // 2D sigma_path[s][v] : number of path "source s to v"
    vector<vector<int>> Source_distance(V,vector<int>(V,-1));   // 2D sigma_level[s][v]: number of path "source s to v"
    vector<vector<float>> Source_delta(V,vector<float>(V,0.0));    // 2D sigma_path[s][v] : number of path "source s to v"
    vector<int> Source_depth(V,0);  //bellman在backward時需要的depth
    vector<int> Source_depth_temp(V,0);  //bellman在backward時需要的depth
    vector<vector<vector<int>>> Source_S(V,vector<vector<int>>(V));     // S is a 3D stack[source][level][node]: [source的Stack][層數][點的ID]
    int max_depth=0;
    vector<float> BC_D2(V,0.0f);

    long long total_predecessor_count = 0; // To accumulate total predecessors

    double time_phase1=0.0;
    double time_phase2=0.0;
    double start_time=0.0;
    double end_time=0.0;
    printf("start\n");
    int times=0;
    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        // if(times==5){
        //     break;
        // }

        if(csr.csrNodesDegree[s]==2 ){
            times++;

            // vector<vector<int>> S(V);               // S is a 2D stack
            // vector<int> sigma(V, 0);               // Sigma array
            // vector<int> dist(V, -1);                // Distance array
            // vector<float> delta(V, 0.0);           // Delta array
            // // vector<int> S_size(V, 0);              // Stack size for each level
            // queue<int> f1, f2;                     // Current and Next frontier
            // vector<vector<int>> predecessors(V);   // Predecessor list

            // vector<vector<int>> Source_path(V,vector<int>(V,0));    // 2D sigma_path[s][v] : number of path "source s to v"
            // vector<vector<int>> Source_distance(V,vector<int>(V,-1));   // 2D sigma_level[s][v]: number of path "source s to v"
            // vector<vector<float>> Source_delta(V,vector<float>(V,0.0));    // 2D sigma_path[s][v] : number of path "source s to v"
            // vector<int> Source_depth(V,0);  //bellman在backward時需要的depth
            // vector<int> Source_depth_temp(V,0);  //bellman在backward時需要的depth
            // vector<vector<vector<int>>> Source_S(V,vector<vector<int>>(V));     // S is a 3D stack[source][level][node]: [source的Stack][層數][點的ID]
            // int max_depth=0;
            // vector<float> BC_D2(V,0.0f);

            sigma.assign(V, 0);   // Reset sigma to size V with all values 0
            dist.assign(V, -1);   // Reset dist to size V with all values -1
            delta.assign(V, 0.0f);   // Reset dist to size V with all values -1
            S.assign(V, vector<int>());
            predecessors.assign(V, vector<int>());

            Source_depth.assign(V, 0);;
            Source_path.assign(V, vector<int>(V,0));  // Reset Successors with empty vectors
            Source_distance.assign(V, vector<int>(V,-1));  // Reset Successors with empty vectors
            Source_S.resize(V, vector<vector<int>>(V,vector<int>()));
            Source_delta.assign(V, vector<float>(V,0.0f));

            printf("csrNodesDegree[%d]: %d\n",s, csr.csrNodesDegree[s]);

            //D2的node的鄰居當Source，traverse完找到path以及level(distance)
            for (int NeighborSource_index = csr.csrV[s]; NeighborSource_index < csr.csrV[s + 1] ; ++NeighborSource_index) {
                int NeighborSourceID = csr.csrE[NeighborSource_index];
                // printf("NeighborSourceID: %d\n",NeighborSourceID);
                Source_path[NeighborSourceID][NeighborSourceID]=1;
                Source_distance[NeighborSourceID][NeighborSourceID]=0;
                f1.push(NeighborSourceID);

                int level = 0;
                //each neighborSource BFS forward phase
                while (!f1.empty()) {
                    while (!f1.empty()) {
                        int traverse_ID = f1.front();
                        f1.pop();
                        Source_S[NeighborSourceID][level].push_back(traverse_ID);

                        // Traverse neighbors in CSR
                        for (int i = csr.csrV[traverse_ID]; i < csr.csrV[traverse_ID + 1]; ++i) {
                            int traverse_Neighbor_ID = csr.csrE[i];

                            if (Source_distance[NeighborSourceID][traverse_Neighbor_ID] < 0) {
                                Source_distance[NeighborSourceID][traverse_Neighbor_ID] = Source_distance[NeighborSourceID][traverse_ID] + 1;
                                f2.push(traverse_Neighbor_ID);
                            }

                            if (Source_distance[NeighborSourceID][traverse_Neighbor_ID] == Source_distance[NeighborSourceID][traverse_ID] + 1) {
                                Source_path[NeighborSourceID][traverse_Neighbor_ID] += Source_path[NeighborSourceID][traverse_ID];
                            }
                        }
                    }
                    swap(f1, f2);
                    level++;
                }
                Source_depth[NeighborSourceID]=(level-1);
                max_depth=max(max_depth,Source_depth[NeighborSourceID]);
                //each neighborSource BFS backward phase
                

                // S.assign(V, vector<int>());  // Reset S with empty vectors

                // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
                //         printf("Source_path[%d]: %d \n",v_ID,Source_path[NeighborSourceID][v_ID]);
                // }

            }
            // printf("max_depth: %d\n",max_depth);
            Source_depth_temp=Source_depth;
            // backward 累加 D2以及兩個D2鄰居的delta BC值
            int b_Source=csr.csrE[csr.csrV[s]];
            int c_Source=csr.csrE[csr.csrV[s]+1];
            while(max_depth > 0){
                // printf("=============depth: %d=============\n",max_depth);
                if(max_depth==Source_depth_temp[b_Source]){ //b側鄰居
                    // printf("do B\n");
                    for (int node : Source_S[b_Source][max_depth]) {
                        for(int neighborIndex = csr.csrV[node]; neighborIndex < csr.csrV[node + 1]; ++neighborIndex){
                            int backneighbor_ID = csr.csrE[neighborIndex];

                            if(Source_distance[b_Source][backneighbor_ID] == Source_distance[b_Source][node] - 1){ //Predecessor
                                
                                Source_delta[b_Source][backneighbor_ID] += (1+Source_delta[b_Source][node])*(Source_path[b_Source][backneighbor_ID])/(float)Source_path[b_Source][node];

                                // printf("b_Source-Source_delta[%d][%d]: %.2f \n",b_Source,backneighbor_ID,Source_delta[b_Source][backneighbor_ID]);
                                if(Source_distance[b_Source][node] == Source_distance[c_Source][node] && node!=s ){
                                    Source_delta[s][backneighbor_ID]+= (1.0f+Source_delta[s][node])*(Source_path[b_Source][backneighbor_ID])/(float)(Source_path[b_Source][node] + Source_path[c_Source][node]);
                                }

                                if(Source_distance[b_Source][node] < Source_distance[c_Source][node]){
                                    // printf("b-Source_delta[%d][%d]: %.2f \n",s,backneighbor_ID,Source_delta[s][backneighbor_ID]);
                                    Source_delta[s][backneighbor_ID] += (Source_path[b_Source][backneighbor_ID])/(float)Source_path[b_Source][node]*(1.0f+Source_delta[s][node]);
                                    // printf("-Source_delta[%d][%d]: %.2f \n",s,node,Source_delta[s][node]);
                                    // printf("-Source_path[%d][%d] : %d \n",b_Source,node,Source_path[b_Source][node]);
                                    // printf("-Source_path[%d][%d] : %d \n",b_Source,backneighbor_ID,Source_path[b_Source][backneighbor_ID]);
                                    // printf("a-Source_delta[%d][%d]: %.2f \n",s,backneighbor_ID,Source_delta[s][backneighbor_ID]);
                                }

                            }
                        }
                        
                    }
                    Source_depth_temp[b_Source]--;
                }
                if(max_depth==Source_depth_temp[c_Source]){ //c側鄰居
                    // printf("do C\n");
                    for (int node : Source_S[c_Source][max_depth]) {
                        for(int neighborIndex = csr.csrV[node]; neighborIndex < csr.csrV[node + 1]; ++neighborIndex){
                            int backneighbor_ID = csr.csrE[neighborIndex];
                            if(Source_distance[c_Source][backneighbor_ID] == Source_distance[c_Source][node] - 1){ //Predecessor
                                Source_delta[c_Source][backneighbor_ID] += (1+Source_delta[c_Source][node])*(Source_path[c_Source][backneighbor_ID])/(float)Source_path[c_Source][node];

                                if(Source_distance[b_Source][node] == Source_distance[c_Source][node] && node!=s){
                                    Source_delta[s][backneighbor_ID]+= (1+Source_delta[s][node])*(Source_path[c_Source][backneighbor_ID])/(float)(Source_path[b_Source][node] + Source_path[c_Source][node]);
                                }

                                if(Source_distance[c_Source][node] < Source_distance[b_Source][node]){
                                    Source_delta[s][backneighbor_ID] += (1+Source_delta[s][node])*(Source_path[c_Source][backneighbor_ID])/(float)Source_path[c_Source][node];
                                    // printf("--Source_delta[%d][%d]: %.2f \n",s,node,Source_delta[s][node]);
                                    // printf("--Source_path[%d][%d] : %d \n",c_Source,node,Source_path[c_Source][node]);
                                    // printf("--Source_path[%d][%d] : %d \n",c_Source,backneighbor_ID,Source_path[c_Source][backneighbor_ID]);
                                    // printf("--Source_delta[%d][%d]: %.2f \n",s,backneighbor_ID,Source_delta[s][backneighbor_ID]);
                                }

                            }
                        }

                    }


                    Source_depth_temp[c_Source]--;
                }
                
                // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
                //     printf("Source_delta[%d][%d]: %.2f \n",b_Source,v_ID,Source_delta[b_Source][v_ID]);
                // }
                // printf("----------\n");
                // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
                //     printf("Source_delta[%d][%d]: %.2f \n",c_Source,v_ID,Source_delta[c_Source][v_ID]);
                // }
                // printf("----------\n");
                // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
                //     printf("Source_delta[%d][%d]: %.2f \n",s,v_ID,Source_delta[s][v_ID]);
                // }
                max_depth--;
            }

            
            // for(int vertexID = csr.startNodeID; vertexID <= csr.endNodeID; ++vertexID){
            //     if(vertexID != ){

            //     }
            //     BC_D2[vertexID]+= Source_delta[s][backneighbor_ID];
            // }


            //透過前面D2鄰居的sigma以及level可以推敲出 D2node的path數量 (檢測path是否正確)
            // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
            //     if(v_ID==s){
            //         Source_distance[s][s]=0;
            //         Source_path[s][s]=1;
            //     }else{
            //         if( Source_distance[csr.csrE[csr.csrV[s]]][v_ID] == Source_distance[csr.csrE[csr.csrV[s]+1]][v_ID]){
            //             Source_distance[s][v_ID]=Source_distance[csr.csrE[csr.csrV[s]]][v_ID]+1;
            //             Source_path[s][v_ID]=Source_path[csr.csrE[csr.csrV[s]]][v_ID]+Source_path[csr.csrE[csr.csrV[s]+1]][v_ID];
            //         }else if( Source_distance[csr.csrE[csr.csrV[s]]][v_ID] < Source_distance[csr.csrE[csr.csrV[s]+1]][v_ID] ){
            //             Source_path[s][v_ID]=Source_path[csr.csrE[csr.csrV[s]]][v_ID];
            //             Source_distance[s][v_ID]=Source_distance[csr.csrE[csr.csrV[s]]][v_ID]+1;
            //         }else{
            //             Source_path[s][v_ID]=Source_path[csr.csrE[csr.csrV[s]+1]][v_ID];
            //             Source_distance[s][v_ID]=Source_distance[csr.csrE[csr.csrV[s]+1]][v_ID]+1;
            //         }
            //     }
            // }


           

            #pragma region make_ans
            //************************************************ */
            //                      對答案程式
            //************************************************ */
            //使用原本的D2為Source的path 以及distance檢查是否正確。
            sigma[s] = 1;
            dist[s] = 0;
            f1.push(s);
            int level = 0;
            // BFS forward phase
            while (!f1.empty()) {
                while (!f1.empty()) {
                    int traverse_ID = f1.front();
                    f1.pop();
                    S[level].push_back(traverse_ID);
                    // Traverse neighbors in CSR
                    for (int i = csr.csrV[traverse_ID]; i < csr.csrV[traverse_ID + 1]; ++i) {
                        int traverse_Neighbor_ID = csr.csrE[i];
                        if (dist[traverse_Neighbor_ID] < 0) {
                            dist[traverse_Neighbor_ID] = dist[traverse_ID] + 1;
                            f2.push(traverse_Neighbor_ID);
                        }
                        if (dist[traverse_Neighbor_ID] == dist[traverse_ID] + 1) {
                            sigma[traverse_Neighbor_ID] += sigma[traverse_ID];
                            predecessors[traverse_Neighbor_ID].push_back(traverse_ID);
                        }
                    }
                }
                swap(f1, f2);
                level++;
            }

            //正確答案的backward
            for (int d = level - 1; d >= 0; --d) {
                for (int w : S[d]) {
                    for (int v : predecessors[w]) {
                        if(v!=s){
                            delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
                        }  
                    }
                    if (w != s) {
                        BC_ans[w] += delta[w];
                    }
                }
            }
             #pragma endregion


            // bool flag=true;
            // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
            //     if(sigma[v_ID]!=Source_path[s][v_ID]){
            //         flag=false;
            //         printf("sigma[%d]: %d  Source_path[s][%d]: %d\n",v_ID,sigma[v_ID],v_ID,Source_path[s][v_ID]);
            //     }
            // }
            // if(flag)
            //     printf("[Correct] Same path!!\n");

            bool flag = true;
            for (int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID) {
                // 取小數點後兩位，並捨棄後續位數
                float delta_rounded = std::floor(delta[v_ID] * 1000) / 1000;
                float source_delta_rounded = std::floor(Source_delta[s][v_ID] * 1000) / 1000;

                if (delta_rounded != source_delta_rounded) {
                    flag = false;
                    printf("delta[%d]: %.3f  delta[%d][%d]: %.3f\n",
                           v_ID, delta_rounded, s, v_ID, source_delta_rounded);
                }
            }
            if (flag)
                printf("[Correct] Same BC from D2!!\n");

           
            // break;
        }
        
        
    }
    // printf("phase1 time: %0.6f\n", time_phase1);
    // printf("phase2 time: %0.6f\n", time_phase2);
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


void bellman_all_Degree(CSR& csr, int V, float* BC_ans) {
    // Allocate memory for BFS data structures
    vector<vector<int>> S(V);               // S is a 2D stack
    vector<int> sigma(V, 0);               // Sigma array
    vector<int> dist(V, -1);                // Distance array
    vector<float> delta(V, 0.0);           // Delta array
    // vector<int> S_size(V, 0);              // Stack size for each level
    queue<int> f1, f2;                     // Current and Next frontier
    vector<vector<int>> predecessors(V);   // Predecessor list

    vector<vector<int>> Source_path(V,vector<int>(V,0));    // 2D sigma_path[s][v] : number of path "source s to v"
    vector<vector<int>> Source_distance(V,vector<int>(V,-1));   // 2D sigma_level[s][v]: number of path "source s to v"
    vector<vector<float>> Source_delta(V,vector<float>(V,0.0));    // 2D sigma_path[s][v] : number of path "source s to v"
    vector<int> Source_depth(V,0);  //bellman在backward時需要的depth
    vector<int> Source_depth_temp(V,0);  //bellman在backward時需要的depth
    vector<vector<vector<int>>> Source_S(V,vector<vector<int>>(V));     // S is a 3D stack[source][level][node]: [source的Stack][層數][點的ID]
    int max_depth=0;
    vector<float> BC_D2(V,0.0f);

    long long total_predecessor_count = 0; // To accumulate total predecessors

    double time_phase1=0.0;
    double time_phase2=0.0;
    double start_time=0.0;
    double end_time=0.0;
    printf("start\n");
    int times=0;
    for (int s = csr.startNodeID; s <= csr.endNodeID; ++s) {
        if(times==2){
            break;
        }

        if(csr.csrNodesDegree[s]==3 ){
            times++;

            sigma.assign(V, 0);   // Reset sigma to size V with all values 0
            dist.assign(V, -1);   // Reset dist to size V with all values -1
            delta.assign(V, 0.0f);   // Reset dist to size V with all values -1
            S.assign(V, vector<int>());
            predecessors.assign(V, vector<int>());

            Source_depth.assign(V, 0);;
            Source_path.assign(V, vector<int>(V,0));  // Reset Successors with empty vectors
            Source_distance.assign(V, vector<int>(V,-1));  // Reset Successors with empty vectors
            Source_S.resize(V, vector<vector<int>>(V,vector<int>()));
            Source_delta.assign(V, vector<float>(V,0.0f));

            printf("csrNodesDegree[%d]: %d\n",s, csr.csrNodesDegree[s]);

            //D2的node的鄰居當Source，traverse完找到path以及level(distance)
            for (int NeighborSource_index = csr.csrV[s]; NeighborSource_index < csr.csrV[s + 1] ; ++NeighborSource_index) {
                int NeighborSourceID = csr.csrE[NeighborSource_index];
                // printf("NeighborSourceID: %d\n",NeighborSourceID);
                Source_path[NeighborSourceID][NeighborSourceID]=1;
                Source_distance[NeighborSourceID][NeighborSourceID]=0;
                f1.push(NeighborSourceID);

                int level = 0;
                //each neighborSource BFS forward phase
                while (!f1.empty()) {
                    while (!f1.empty()) {
                        int traverse_ID = f1.front();
                        f1.pop();
                        Source_S[NeighborSourceID][level].push_back(traverse_ID);

                        // Traverse neighbors in CSR
                        for (int i = csr.csrV[traverse_ID]; i < csr.csrV[traverse_ID + 1]; ++i) {
                            int traverse_Neighbor_ID = csr.csrE[i];

                            if (Source_distance[NeighborSourceID][traverse_Neighbor_ID] < 0) {
                                Source_distance[NeighborSourceID][traverse_Neighbor_ID] = Source_distance[NeighborSourceID][traverse_ID] + 1;
                                f2.push(traverse_Neighbor_ID);
                            }

                            if (Source_distance[NeighborSourceID][traverse_Neighbor_ID] == Source_distance[NeighborSourceID][traverse_ID] + 1) {
                                Source_path[NeighborSourceID][traverse_Neighbor_ID] += Source_path[NeighborSourceID][traverse_ID];
                            }
                        }
                    }
                    swap(f1, f2);
                    level++;
                }
                Source_depth[NeighborSourceID]=(level-1);
                max_depth=max(max_depth,Source_depth[NeighborSourceID]);
                //each neighborSource BFS backward phase
                

                // S.assign(V, vector<int>());  // Reset S with empty vectors

                // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
                //         printf("Source_path[%d]: %d \n",v_ID,Source_path[NeighborSourceID][v_ID]);
                // }

            }
            // printf("max_depth: %d\n",max_depth);
            Source_depth_temp=Source_depth;
            // backward 累加 D2以及兩個D2鄰居的delta BC值
            vector<int> s_N(csr.csrNodesDegree[s]);
            for (int i = 0; i < csr.csrNodesDegree[s]; ++i) {
                s_N[i] = csr.csrE[csr.csrV[s] + i];
            }
            
            while (max_depth > 0) {
                // 遍歷所有鄰居
                for (int i = 0; i < csr.csrNodesDegree[s]; ++i) {
                    int current_source = s_N[i];
            
                    if (max_depth == Source_depth_temp[current_source]) {
                        // 處理當前 source 的節點
                        for (int node : Source_S[current_source][max_depth]) {
                            for (int neighborIndex = csr.csrV[node]; neighborIndex < csr.csrV[node + 1]; ++neighborIndex) {
                                int backneighbor_ID = csr.csrE[neighborIndex];
            
                                if (Source_distance[current_source][backneighbor_ID] == Source_distance[current_source][node] - 1) { // Predecessor
                                    Source_delta[current_source][backneighbor_ID] += (1.0f + Source_delta[current_source][node]) *
                                            (Source_path[current_source][backneighbor_ID]) / (float)Source_path[current_source][node];
                                    
                                    // 確保 Source_distance[current_source][node] 是最小的
                                    bool is_min_distance = true;
                                    for (int j = 0; j < csr.csrNodesDegree[s]; ++j) {
                                        if (Source_distance[s_N[j]][node] < Source_distance[current_source][node]) {
                                            is_min_distance = false;
                                            break;
                                        }
                                    }
            
                                    if (is_min_distance && s!=node) {
                                        // 累加所有最小距離的路徑數
                                        int total_min_path = 0;
                                        for (int j = 0; j < csr.csrNodesDegree[s]; ++j) {
                                            if (Source_distance[s_N[j]][node] == Source_distance[current_source][node]) {
                                                total_min_path += Source_path[s_N[j]][node];
                                            }
                                        }
                                        
                                        Source_delta[s][backneighbor_ID] += (1.0f + Source_delta[s][node])/ (float)total_min_path;
                                        
                                    }

                                }
                            }
                        }
                        Source_depth_temp[current_source]--;
                    }
                }
                max_depth--;
            }

            

            //透過前面D2鄰居的sigma以及level可以推敲出 D2node的path數量 (檢測path是否正確)
            // for(int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID){
            //     if(v_ID==s){
            //         Source_distance[s][s]=0;
            //         Source_path[s][s]=1;
            //     }else{
            //         if( Source_distance[csr.csrE[csr.csrV[s]]][v_ID] == Source_distance[csr.csrE[csr.csrV[s]+1]][v_ID]){
            //             Source_distance[s][v_ID]=Source_distance[csr.csrE[csr.csrV[s]]][v_ID]+1;
            //             Source_path[s][v_ID]=Source_path[csr.csrE[csr.csrV[s]]][v_ID]+Source_path[csr.csrE[csr.csrV[s]+1]][v_ID];
            //         }else if( Source_distance[csr.csrE[csr.csrV[s]]][v_ID] < Source_distance[csr.csrE[csr.csrV[s]+1]][v_ID] ){
            //             Source_path[s][v_ID]=Source_path[csr.csrE[csr.csrV[s]]][v_ID];
            //             Source_distance[s][v_ID]=Source_distance[csr.csrE[csr.csrV[s]]][v_ID]+1;
            //         }else{
            //             Source_path[s][v_ID]=Source_path[csr.csrE[csr.csrV[s]+1]][v_ID];
            //             Source_distance[s][v_ID]=Source_distance[csr.csrE[csr.csrV[s]+1]][v_ID]+1;
            //         }
            //     }
            // }


           

            #pragma region make_ans
            //************************************************ */
            //                      對答案程式
            //************************************************ */
            //使用原本的D2為Source的path 以及distance檢查是否正確。
            sigma[s] = 1;
            dist[s] = 0;
            f1.push(s);
            int level = 0;
            // BFS forward phase
            while (!f1.empty()) {
                while (!f1.empty()) {
                    int traverse_ID = f1.front();
                    f1.pop();
                    S[level].push_back(traverse_ID);
                    // Traverse neighbors in CSR
                    for (int i = csr.csrV[traverse_ID]; i < csr.csrV[traverse_ID + 1]; ++i) {
                        int traverse_Neighbor_ID = csr.csrE[i];
                        if (dist[traverse_Neighbor_ID] < 0) {
                            dist[traverse_Neighbor_ID] = dist[traverse_ID] + 1;
                            f2.push(traverse_Neighbor_ID);
                        }
                        if (dist[traverse_Neighbor_ID] == dist[traverse_ID] + 1) {
                            sigma[traverse_Neighbor_ID] += sigma[traverse_ID];
                            predecessors[traverse_Neighbor_ID].push_back(traverse_ID);
                        }
                    }
                }
                swap(f1, f2);
                level++;
            }

            //正確答案的backward
            for (int d = level - 1; d >= 0; --d) {
                for (int w : S[d]) {
                    for (int v : predecessors[w]) {
                        if(v!=s){
                            delta[v] += (sigma[v] / (float)sigma[w]) * (1.0 + delta[w]);
                        }  
                    }
                    if (w != s) {
                        BC_ans[w] += delta[w];
                    }
                }
            }
             #pragma endregion

            bool flag = true;
            for (int v_ID = csr.startNodeID; v_ID <= csr.endNodeID; ++v_ID) {
                // 取小數點後兩位，並捨棄後續位數
                float delta_rounded = std::floor(delta[v_ID] * 1000) / 1000;
                float source_delta_rounded = std::floor(Source_delta[s][v_ID] * 1000) / 1000;

                if (delta_rounded != source_delta_rounded) {
                    flag = false;
                    printf("delta[%d]: %.3f  delta[%d][%d]: %.3f\n",
                           v_ID, delta_rounded, s, v_ID, source_delta_rounded);
                }
            }
            if (flag)
                printf("[Correct] Same BC from D2!!\n");

           
            // break;
        }
        
        
    }
    
}

