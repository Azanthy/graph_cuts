//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_GRAPH_HH
#define GRAPH_CUTS_GRAPH_HH

#include <string>
#include <vector>
#include <memory>

class Graph {
public:
    Graph(char *img, char *seeds);
    void max_flow();
    void push(int x, int y);
    void relabel(int x, int y);
    bool is_active(int x, int y);
    bool any_active();
    void print();

private:
    void normalize_histo(size_t *bck_histo, size_t *obj_histo, float **norm_bck_histo, float **norm_obj_histo, float *sum_bck_histo, float *sum_obj_histo);
    void initialize_node_capacities(int x, int y, std::vector<int> &grays);


    int _width;
    int _height;
    int _size;
    unsigned int _nb_active;

    std::vector<int> _heights;
    std::vector<float> _excess_flow;
    std::vector<float> _up;
    std::vector<float> _right;
    std::vector<float> _bottom;
    std::vector<float> _left;

    std::vector<float> _neighbors[4];    // {up, right, bottom, left}
    const int x_nghb[4] = {0, 1, 0, -1}; // idx offset for x axis
    const int y_nghb[4] = {-1, 0, 1, 0}; // idx offset for y axis
    const int id_opp[4] = {2, 3, 0, 1};  // idx of opposite array up=0, right=1, bottom=2, left=3

};


#endif //GRAPH_CUTS_GRAPH_HH
