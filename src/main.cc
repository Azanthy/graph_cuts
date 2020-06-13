#include <iostream>
#include <unistd.h>
#include "graph.hh"
#include "gpu.hh"

int parse_options(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, ":cg")) != -1)
    {
        switch (opt)
        {
            case 'c':
                return 0;
            case 'g':
                return 1;
            default:
                return -1;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cerr << "graph_cuts: should be ./graph_cut [-cg] <img> <seeds_img>" << std::endl;
        exit(-1);
    }
    int algo = parse_options(argc, argv);
    auto graph = Graph(argv[optind], argv[optind+1], 400);
    if (!algo)
        graph.max_flow();
    else
        max_flow_gpu(graph);
    graph.dfs();
    graph.print();
    if (!algo)
        graph.free();
}
