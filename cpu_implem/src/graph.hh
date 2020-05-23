//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_GRAPH_HH
#define GRAPH_CUTS_GRAPH_HH

#include <string>
#include <vector>
#include <memory>
#include "node.hh"

using shared_node = std::shared_ptr<Node>;

class Graph {
public:
    Graph(char *img, char *seeds);

private:
    void normalize_histo(size_t *bck_histo, size_t *obj_histo, float **norm_bck_histo, float **norm_obj_histo, float *sum_bck_histo, float *sum_obj_histo);
    void initialize_node_capacities(int x, int y, shared_node node);


    std::vector<shared_node> _nodes;
    int _width;
    int _height;

    /* NEXT STEP
    float *excess_flow;
    float *heights; x2
    float *up;
    float *right;
    float *bottom;
    float *left;
    */

};


#endif //GRAPH_CUTS_GRAPH_HH
