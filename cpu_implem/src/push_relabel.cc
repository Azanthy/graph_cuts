#include <string>
#include "push_relabel.hh"

bool is_valid(int i, int j, int h, int w)
{
    return i >= 0 && j >= 0 && i < w && j < h;
}

Algo::Algo(string normal_file, string marked_file)
{
    //ici récup les images ou alors avant mais récup la height et weight

    left = std::vector<flow_type>(width * height, -1);
    top = std::vector<flow_type>(width * height, -1);
    right = std::vector<flow_type>(width * height, -1);
    bottom = std::vector<flow_type>(width * height, -1);
    //ici modif les 4 vecteurs avec les capacity entre pixels voisins
    height_vect = std::vector<int>(width * height, 0);
    flow = std::vector<flow_type>(width * height);
    //ici modif flow avec la soustraction entre les poids vers la source et le sink
    max_height = width * height;
    h = height;
    w = width;
}

void Algo::push_single_neighbour(std::vector<flow_type> &to_neighbour,
                                    std::vector<flow_type> &from_neighbour,
                                    int i, int j, int x, int y)
{
    int idx = j * w + i;
    if (is_valid(i + x, j + y, h, w))
    {
        int neighbour = (j+y) * w + i + x;
        if (height_vect[neighbour] == height_vect[idx] - 1)
        {
            flow_type f = std::min(to_neighbour[idx], flow[idx]);
            flow[idx] -= f;
            flow[neighbour] += f;
            to_neighbour[idx] -= f;
            from_neighbour[neighbour] += f;
        }
    }
}

Algo::push(int i, int j)
{
    int idx = j * w + i;
    if (flow[idx] > 0 && height_vect[idx] < max_height)
    {
        push_single_neighbour(top, bottom, i, j, 0, -1);
        push_single_neighbour(bottom, top, i, j, 0, 1);
        push_single_neighbour(left, right, i, j, -1, 0);
        push_single_neighbour(right, left, i, j, 1, 0);
    }
}

int Algo::relabel_single_neighbour(int i, int j, int x, int y, int cur_height
        std::vector<flow_type> &to_neighbour)
{
    int idx = j * w + i;
    if (is_valid(i + x, j + y, h, w))
    {
        int neighbour = (j+y) * w + i + x;
        if (to_neighbour[idx] > 0)
            cur_height = std::min(cur_height, height_vect[neighbour]+1);
    }
    return cur_height;
}

void Algo::relabel(int i, int j)
{
    int idx = j * w + i;
    if (flow[idx] > 0 && height_vect[idx] < max_height)
    {
        int cur_height = max_height;
        cur_height = relabel_single_neighbour(i, j, 0, -1, cur_height, top);
        cur_height = relabel_single_neighbour(i, j, 0, 1, cur_height, bottom);
        cur_height = relabel_single_neighbour(i, j, -1, 0, cur_height, left);
        cur_height = relabel_single_neighbour(i, j, 1, 0, cur_height, right);
        height_vect[idx] = cur_height;
    }
}
