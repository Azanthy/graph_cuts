//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_NODE_HH
#define GRAPH_CUTS_NODE_HH


class Node {
public:
    Node(char gray_val)
        : _gray_val(gray_val)
        , _excess_flow(0.f)
        , _height(0)
    {};

    int _height;
    char _gray_val;
    float _excess_flow;
    float _capacities[4] = {0};  // [up, right, bottom, left]

    // Neighbor;

};


#endif //GRAPH_CUTS_NODE_HH
