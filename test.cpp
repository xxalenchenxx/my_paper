#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>

using namespace std;

// Bottom-up BFS step function
void bottom_up_step(const vector<vector<int>>& graph, const unordered_set<int>& frontier, unordered_set<int>& next, vector<int>& parents) {
    int num_vertices = graph.size();

    for (int v = 0; v < num_vertices; ++v) {
        // Check if the node v has not been visited (parents[v] == -1)
        if (parents[v] == -1) {
            // Iterate through all neighbors of vertex v
            for (int n : graph[v]) {
                // If a neighbor n is in the current frontier
                if (frontier.find(n) != frontier.end()) {
                    // Update parent of v and add v to the next frontier
                    parents[v] = n;
                    next.insert(v);
                    break; // No need to check more neighbors
                }
            }
        }
    }
}

void bfs_bottom_up(const vector<vector<int>>& graph, int start) {
    int num_vertices = graph.size();
    vector<int> parents(num_vertices, -1); // Parent array, -1 means unvisited
    unordered_set<int> frontier; // Current frontier
    unordered_set<int> next; // Next frontier

    // Initialize with the start node
    frontier.insert(start);
    parents[start] = start; // Root node points to itself

    // Bottom-up BFS loop
    while (!frontier.empty()) {
        next.clear();

        // Perform one bottom-up step
        bottom_up_step(graph, frontier, next, parents);

        // Move to the next level
        frontier = next;
    }

    // Output the result
    cout << "Parents array (showing the BFS tree):" << endl;
    for (int i = 0; i < num_vertices; ++i) {
        cout << "Node " << i << " -> Parent: " << parents[i] << endl;
    }
}

int main() {
    // Example graph (adjacency list)
    vector<vector<int>> graph = {
        {1, 2},      // Neighbors of node 0
        {0,1, 3, 4},   // Neighbors of node 1
        {0,2, 5, 6},   // Neighbors of node 2
        {1},         // Neighbors of node 3
        {1},         // Neighbors of node 4
        {2},         // Neighbors of node 5
        {2}          // Neighbors of node 6
    };

    int start_node = 0;
    bfs_bottom_up(graph, start_node);

    return 0;
}
