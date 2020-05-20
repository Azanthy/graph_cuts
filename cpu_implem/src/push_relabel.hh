#include <string>
#include <vector>
#include <algorithm>

typedef int flow_type

class Algo
{
public:
    Algo(string normal_file, string marked_file);
    void push_single_neighbour(std::vector<flow_type> &to_neighbour,
                                  std::vector<flow_type> &from_neighbour,
                                  int i, int j, int x, int y);
    int relabel_single_neighbour(int i, int j, int x, int y, int cur_height
            std::vector<flow_type> &to_neighbour);
    void relabel(int i, int j);
    void push(int i, int j);
private:
    std::vector<flow_type> left;
    std::vector<flow_type> top;
    std::vector<flow_type> right;
    std::vector<flow_type> bottom;
    std::vector<int> height_vect;
    std::vector<flow_type> flow;
    int max_height;
    int h;
    int w;
};
