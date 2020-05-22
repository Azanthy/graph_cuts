//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_GRAPH_HH
#define GRAPH_CUTS_GRAPH_HH

#include <string>
#include <vector>
#include "node.hh"

class Graph {
public:
    Graph(char *img, char *seeds);

private:
    std::vector<Node> nodes;
    int width_;
    int height_;
    size_t bck_histo[256];
    size_t obj_histo[64];

};


#endif //GRAPH_CUTS_GRAPH_HH
