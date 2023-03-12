#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <map>

#define NODE_NUM 1850065 // Number of nodes in the graph. 1849272

#define ALPHA 0.2        // Dumping factor
#define EPSILON 0.000001 // Very small number for convergence

#define MAX_THREAD_NUMS 8 // Maximum number of threads to test.

#define GRAPH "graph.txt"
#define CSV_FILE_NAME "results.csv"

using namespace std;

int main()
{
    vector<int> row_begin;
    vector<double> values;
    vector<int> col_indices;

    vector<pair<string, string>> graph;

    int nodeIndex = 0;            // To use it to assign indexes to nodes.
    vector<int> out(NODE_NUM);    // Outgoing Degrees of the nodes.
    map<string, int> nodeIndexes; // Hash map of names and indexes of the nodes.

    for (int i = 0; i < NODE_NUM; i++)
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
    vector<double> ranks(NODE_NUM);    // r^t
    vector<double> newRanks(NODE_NUM); // r^t+1

    vector<pair<omp_sched_t, string>> schedules = {// To iterate through different types of schedules.
        make_pair(omp_sched_guided, "GUIDED"),
        make_pair(omp_sched_static, "STATIC"),
        make_pair(omp_sched_dynamic, "DYNAMIC")
    };

    vector<int> chunkSizes = {1000, 5000, 10000};

    ofstream csvFile;

    csvFile.open(CSV_FILE_NAME);

    csvFile << "Test No., Scheduling Method, Chunk Size, No. of Iterations, Timings in secs for each number of threads" << endl;
    csvFile << " , , , , ";

    for (int i = 1; i <= MAX_THREAD_NUMS; i++) // Writes thread numbers to the csv file.
    {
        csvFile << i << ", ";
    }

    csvFile << endl;

    int testNo = 1;
    omp_set_dynamic(0); // Calculation will be done in static thread count.

    for (auto schedule : schedules)
    {
        for (int chunkSize : chunkSizes)
        {
            csvFile << testNo << ", " << schedule.second << ", " << chunkSize << ", "; // Writes test details to the csv file.
            cout << "Test #" << testNo << " started." << endl;
            for (int threadCount = 1; threadCount <= MAX_THREAD_NUMS; threadCount++)
            {
                for (double r : ranks)
                {
                    ranks.at(r) = 1.0; // Sets r^t values to 1 before calculation for each test.
                }

                omp_set_schedule(schedule.first, chunkSize); // Sets schedule type and chunk size for each test.
                omp_set_num_threads(threadCount);            // Sets thread count for each test.

                double start = omp_get_wtime(); // Start time

                while (true)
                {
                    iterationCount++;
                    diff = 0.0;
                    #pragma omp parallel shared(ranks, newRanks, row_begin, col_indices, values, diff, iterationCount)
                    {
                        #pragma omp for schedule(runtime)
                        for (int i = 0; i < (int)row_begin.size() - 1; i++)
                        {
                            newRanks.at(i) = 0.0; // r^t+1 given 0 before each calculation

                            for (int j = row_begin.at(i); j < row_begin.at(i + 1); j++) // Calculating ranks according to the formula.
                            {
                                int index = col_indices.at(j);
                                newRanks.at(i) += (ALPHA * values.at(j) * ranks.at(index)) + (1.0 - ALPHA);
                            }
                        }

                        for (int i = 0; i < NODE_NUM; i++) // Calculating L1 Norm
                        { 
                            diff = diff + fabs(newRanks.at(i) - ranks.at(i));
                        }
                    }

                    ranks = newRanks; // ranks are assigned to the newly calculated ranks.
                    if (diff < EPSILON)
                        break; // Finishes iteration when L1 Norm is smaller than Epsilon value.
                }

                double end = omp_get_wtime(); // End Time

                double timePassed = end - start; // Time passed in calculation

                if (threadCount == 1)
                    csvFile << iterationCount << ", "; // Writes iteration count to the csvFile for the first thread of each test.
                
                csvFile << timePassed << ", ";         // Writes the time passed while calculating ranks for each thread.

                iterationCount = 0; // Makes iteration count zero before the next test.
            }
            cout << "Test #" << testNo << " ended." << endl;
            testNo++;
            csvFile << endl; // End of the row of this test.
        }
    }

    cout << "Highest Ranked 5 Nodes: " << endl;

    for (int i = 0; i < 5; i++)
    {
        int index = 0;
        double rank = 0;
        string node;

        for (int j = 0; j < NODE_NUM; j++) // Finds highest ranked node index.
        {
            double r = ranks.at(j);
            if (r > rank)
            {
                index = j;
                rank = r;
            }
        }

        ranks.at(index) = 0.0;                                          // Node will not be considered in the next iteration
        for (auto i = nodeIndexes.begin(); i != nodeIndexes.end(); i++) // Finds the name of the node with the index
        {
            if (i->second == index)
            {
                node = i->first;
                break;
            }
        }
        cout << i + 1 << "- " << node << " -- " << rank << endl;
    }

    return 1;
}