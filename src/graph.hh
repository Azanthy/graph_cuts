//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_GRAPH_HH
#define GRAPH_CUTS_GRAPH_HH

#include <string>
#include <vector>
#include <memory>
#include <queue>

class Graph {
public:
    Graph(char *img, char *seeds);
    ~Graph();
    void max_flow();
    void push(int x, int y);
    void relabel(int x, int y, int *heights);
    bool is_active(int x, int y);
    bool any_active();
    void dfs();
    void print();

    float max_capacity(int x, int y);
    float gradient(int id1, int id2);
    float gradient(int id, int mean[]);
    void normalize_histo(size_t **bck_histo, size_t **obj_histo, float **norm_bck_histo, float **norm_obj_histo);
    void initialize_node_capacities(int x, int y);


    int _width;
    int _height;
    int _size;
    unsigned int _nb_active;
    unsigned char *_img;
    unsigned char *_labels;

    std::queue<int> _dfs;
    std::vector<bool> _binary;

    int *_heights;
    int *_excess_flow;
    int *_up;
    int *_right;
    int *_bottom;
    int *_left;

    int *_neighbors[4];    // {up, right, bottom, left}
    const int x_nghb[4] = {0, 1, 0, -1}; // idx offset for x axis
    const int y_nghb[4] = {-1, 0, 1, 0}; // idx offset for y axis
    const int id_opp[4] = {2, 3, 0, 1};  // idx of opposite array up=0, right=1, bottom=2, left=3

};


#endif //GRAPH_CUTS_GRAPH_HH
