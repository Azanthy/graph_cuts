#include <iostream>

#include "graph.hh"

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "graph_cuts: should be ./graph_cut <img> <seeds_img>" << std::endl;
        exit(-1);
    }
    auto graph = Graph(argv[1], argv[2]);
}
