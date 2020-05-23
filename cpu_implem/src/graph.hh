//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_GRAPH_HH
#define GRAPH_CUTS_GRAPH_HH

#include <string>
#include <vector>
#include <memory>
#include "node.hh"

class Graph {
public:
    Graph(char *img, char *seeds);

private:
    void normalize_histo(size_t *bck_histo, size_t *obj_histo, float **norm_bck_histo, float **norm_obj_histo, float *sum_bck_histo, float *sum_obj_histo);


    std::vector<std::shared_ptr<Node>> _nodes;
    int _width;
    int _height;

};


#endif //GRAPH_CUTS_GRAPH_HH
