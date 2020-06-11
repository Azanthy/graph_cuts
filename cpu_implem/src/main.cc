#include <iostream>

#include "graph.hh"
#include "gpu.hh"

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "graph_cuts: should be ./graph_cut <img> <seeds_img>" << std::endl;
        exit(-1);
    }
    auto graph = Graph(argv[1], argv[2]);
    max_flow_gpu(graph);
    //graph.max_flow();
    //graph.dfs();
    //graph.print();
}
