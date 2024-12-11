#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

// 隨機生成無向圖
void generateGraph(int numNodes, int avgEdges, const string& folderPath, const string& fileName) {
    int numEdges = avgEdges * numNodes / 2; // 總邊數
    vector<set<int>> adjacencyList(numNodes); // 鄰接表
    srand(static_cast<unsigned>((0))); // 隨機種子

    // 確保資料夾存在
    mkdir(folderPath.c_str(), 0777);

    // 隨機生成邊
    for (int i = 0; i < numEdges; ++i) {
        int u = rand() % numNodes;
        int v = rand() % numNodes;
        while (u == v || adjacencyList[u].count(v)) { // 避免自環和重複邊
            u = rand() % numNodes;
            v = rand() % numNodes;
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
    string folderPath = "../dataset/synthesis";
    vector<int> nodeCounts = {100, 500, 1000, 2000,4000, 6000 ,8000, 10000,15000,20000};
    int avgEdges = 16; // 平均每個節點的邊數

    for (int numNodes : nodeCounts) {
        string fileName = "u_" + to_string(numNodes) + "_" + to_string(avgEdges * numNodes / 2) + ".mtx";
        generateGraph(numNodes, avgEdges, folderPath, fileName);
    }

    return 0;
}
