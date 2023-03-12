#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <map>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/complex.h>
#include <thrust/sort.h>

#define NODE_NUM 1850065 // Number of nodes in the graph. 1849272

#define ALPHA 0.2        // Dumping factor
#define EPSILON 0.000001 // Very small number for convergence

#define GRAPH "graph.txt"

using namespace std;

// fabs function is turned to a functor in order to use it with thrust transform.
struct absoluteValue {
    __host__ __device__
    double operator()(const double& a, const double& b){
        return fabs(a - b);
    }    
};

int main()
{
    thrust::host_vector<int> row_begin;
    thrust::host_vector<double> values;
    thrust::host_vector<int> col_indices;

    vector<pair<string, string>> graph;

    int nodeIndex = 0;            // To use it to assign indexes to nodes.
    vector<int> out(NODE_NUM);    // Outgoing Degrees of the nodes.
    map<string, int> nodeIndexes; // Hash map of names and indexes of the nodes.
    int i = 0;
    for (i = 0; i < NODE_NUM; i++)
    {
        out.at(i) = 0;
    }

    ifstream file(GRAPH);

    cout << "Opening the file..." << endl;
    if (!file.is_open())
    {
        cout << "Failed to open file.\n";
        return 0;
    }

    cout << "File is opened. Reading..." << endl;

    string line;
    string node1;
    string node2;

    while (getline(file, line)) // Stores graph in a vector to iterate in it faster afterwards.
    {
        istringstream ss(line);

        ss >> node1 >> node2;
        graph.emplace_back(node1, node2);
    }
    file.close();

    cout << "File has been read." << endl;
    cout << "Assigning indexes..." << endl;

    for (auto g : graph)
    { // Assignes indexes to the nodes to refer to them afterwards.
        string node1 = g.first;
        string node2 = g.second;

        int node1Index = 0;
        int node2Index = 0;

        auto node1Find = nodeIndexes.find(node1);
        if (node1Find == nodeIndexes.end()) // if node is not given index
        {
            node1Index = nodeIndex;
            nodeIndex++;
            nodeIndexes.emplace(node1, node1Index);
        }
        else
        {
            node1Index = node1Find->second;
        }

        out.at(node1Index)++;

        auto node2Find = nodeIndexes.find(node2);
        if (node2Find == nodeIndexes.end()) // if node is not given index
        {
            node2Index = nodeIndex;
            nodeIndex++;
            nodeIndexes.emplace(node2, node2Index);
        }
    }
    cout << "Indexes are assigned." << endl;

    cout << "Creating CSR Matrix..." << endl;

    string lastNode;
    for (auto g : graph)
    {
        int node1Index = 0;

        string node1 = g.first;
        string node2 = g.second;

        auto node1Find = nodeIndexes.find(node1);

        node1Index = node1Find->second;

        if (node2 != lastNode) // if node2 is not the same with the previous node, this row finishes.
        {
            row_begin.push_back(values.size());
        }

        col_indices.push_back(node1Index);          // column index is equal to index of node1.
        values.push_back(1.0 / out.at(node1Index)); // values are calculated based on outgoing degrees of nodes.

        lastNode = node2;
    }

    row_begin.push_back(values.size()); // For the last row

    cout << "CSR Matrix is created" << endl;

    cout << "Starting rank calculations..." << endl;

    int iterationCount = 0;
    double diff;
    thrust::host_vector<double> ranks(NODE_NUM, 1.0);    // r^t
    thrust::host_vector<double> newRanks(NODE_NUM, 0); // r^t+1
    thrust::host_vector<double> diffs(NODE_NUM); // differences between each rank in r^t, r^t+1
    
    double start = clock(); // Start time

    while (true)
    {
        iterationCount++;
        diff = 0.0;
        for (i = 0; i < (int)row_begin.size() - 1; i++)
        {
            int rowStart = row_begin[i]; 
            int rowEnd = row_begin[i+1];
            // cout << "rowStart: " << rowStart << endl;
            // previousRank = r^t -> through rowStart to rowEnd
            auto previousRank = thrust::make_permutation_iterator(ranks.begin(), col_indices.begin() + rowStart);
            // totalAlpha = In each calculation we need to sum with (1.0 - ALPHA). So we need rowEnd - rowStart time of it.
            double totalAlpha = (1.0 - ALPHA) * (rowEnd - rowStart);
            // totalMultiplication = The total of P*r(t) -> t throught rowStart to rowEnd
            double totalMultiplication = thrust::inner_product(values.begin() + rowStart, values.begin() + rowEnd, previousRank, 0.0);
            newRanks[i] = (ALPHA * totalMultiplication) + totalAlpha;
        }

        thrust::transform(newRanks.begin(), newRanks.end(), ranks.begin(), diffs.begin(), absoluteValue());

        diff = thrust::reduce(diffs.begin(), diffs.end());
        cout << "Difference: " << diff << endl;
        
        ranks = newRanks; // ranks are assigned to the newly calculated ranks.
        if (diff < EPSILON)
            break; // Finishes iteration when L1 Norm is smaller than Epsilon value.
    }

    double end = clock(); // End Time

    double timePassed = end - start; // Time passed in calculation

    cout << "Time passed in calculations: " << timePassed << endl;
    cout << "Iteration Count: " << iterationCount << endl;

    cout << "Highest Ranked 5 Nodes: " << endl;

    for (i = 0; i < 5; i++)
    {
        int index = 0;
        double rank = 0;
        string node;

        for (int j = 0; j < NODE_NUM; j++) // Finds highest ranked node index.
        {
            double r = ranks[j];
            if (r > rank)
            {
                index = j;
                rank = r;
            }
        }

        ranks[index] = 0.0; // Node will not be considered in the next iteration

        for (auto it = nodeIndexes.begin(); it != nodeIndexes.end(); it++) // Finds the name of the node with the index
        {
            if (it->second == index)
            {
                node = it->first;
                break;
            }
        }
        cout << i + 1 << "- " << node << " -- " << rank << endl;
    }

    return 1;
}