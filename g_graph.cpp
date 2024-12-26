#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <random>

using namespace std;

// 隨機生成具有高最大度數的無向圖
void generateGraph(int numNodes, int avgEdges, const string& folderPath, const string& fileName) {
    int numEdges = avgEdges * numNodes / 2; // 總邊數
    vector<set<int>> adjacencyList(numNodes); // 鄰接表
    mt19937 rng(static_cast<unsigned>(0)); // 隨機種子
    uniform_int_distribution<int> nodeDist(0, numNodes - 1);

    // 確保資料夾存在
    mkdir(folderPath.c_str(), 0777);

    // 引入 hub 節點的機制
    int numHubs = max(0, 1); // hub 節點數量為總節點的 5%
    vector<int> hubs(numHubs);

    for (int i = 0; i < numHubs; ++i) {
        hubs[i] = i; // 假設前 numHubs 個節點為 hub
    }

    // 隨機生成邊
    for (int i = 0; i < numEdges; ++i) {
        int u, v;

        if (rng() % 100 < 15 ) { // 30-60% 的邊連接到 hub 節點
            u = hubs[rng() % numHubs];
            v = nodeDist(rng);
        } else { // 其餘的邊為一般隨機連接
            u = nodeDist(rng);
            v = nodeDist(rng);
        }

        while (u == v || adjacencyList[u].count(v)) { // 避免自環和重複邊
            u = nodeDist(rng);
            v = nodeDist(rng);
        }
    
        adjacencyList[u].insert(v);
        adjacencyList[v].insert(u);
    }

    // 輸出到文件
    ofstream outFile(folderPath + "/" + fileName);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open file " << folderPath + "/" + fileName << endl;
        return;
    }

    // 文件頭：節點數和邊數
    outFile << numNodes << " " << numEdges << endl;

    // 輸出邊
    for (int u = 0; u < numNodes; ++u) {
        for (int v : adjacencyList[u]) {
            if (u < v) { // 確保每條邊只輸出一次
                outFile << u << " " << v << endl;
            }
        }
    }

    outFile.close();
    cout << "Graph saved to " << folderPath + "/" + fileName << endl;
}

int main() {
    // 資料夾路徑
    string folderPath = "../dataset/synthesis_high_MAX_avg_16/";
    vector<int> nodeCounts = {100, 500, 1000, 2000, 4000, 6000, 8000};
    int avgEdges = 16; // 平均每個節點的邊數

    for (int numNodes : nodeCounts) {
        string fileName = "u_" + to_string(numNodes) + "_" + to_string(avgEdges * numNodes / 2) + ".mtx";
        generateGraph(numNodes, avgEdges, folderPath, fileName);
    }

    return 0;
}
