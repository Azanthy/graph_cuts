//
// Created by gigi on 5/22/20.
//

#ifndef GRAPH_CUTS_NODE_HH
#define GRAPH_CUTS_NODE_HH


class Node {
public:
    Node(char gray_val)
        : _gray_val(gray_val)
        , _weight_src(-1)
        , _weight_snk(-1)
    {};

    char _gray_val;
    float _weight_snk;
    float _weight_src;

    // Neighbor;

};


#endif //GRAPH_CUTS_NODE_HH
