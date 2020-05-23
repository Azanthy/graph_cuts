//
// Created by gigi on 5/22/20.
//

#include <cmath>
#include <iostream>
#include <fstream>

#include "graph.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define HISTO_FACTOR 4
#define HEIGHT_MAX 5
#define DISTANCE(x,y) std::exp(-std::abs(x - y)/255.f)

Graph::Graph(char *img, char *seeds) {
    // Load image
    int n, tmp_w, tmp_h;
    unsigned char *data = stbi_load(img, &this->_width, &this->_height, &n, 0);
    unsigned char *labels = stbi_load(seeds, &tmp_w, &tmp_h, &n, 0);

    size_t *bck_histo = new size_t[256/HISTO_FACTOR]();
    size_t *obj_histo = new size_t[256/HISTO_FACTOR]();

    this->_size = this->_width * this->_height;
    this->_heights = std::vector<int>(this->_size, 0);
    for (auto i=0; i <4; i++)
        this->_neighbors[i] = std::vector<float>(this->_size,0.f);
    this->_excess_flow = std::vector<float>(this->_size,0.f);

    auto grays = std::vector<int>(this->_size, 0);

    // Initialize nodes and histograms
    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto idx = y * this->_width + x;
            auto n_idx = y * this->_width * n + x * n;
            char gray = 0.299 * data[n_idx] + 0.587 * data[n_idx+1] + 0.114 * data[n_idx+2];

            grays[idx] = gray;
            // Histograms of size 256/4=64
            if (labels[idx]) {
                bck_histo[gray/HISTO_FACTOR] += 1;
                this->_excess_flow[idx] = -1.f;
            }
            if (labels[idx+2]) {
                obj_histo[gray/HISTO_FACTOR] += 1;
                this->_excess_flow[idx] = 1.f;
            }
        }
    }

    // COmpute histograms for external capacities
    float *norm_bck_histo, *norm_obj_histo;
    float sum_bck_histo = 0;
    float sum_obj_histo = 0;
    normalize_histo(bck_histo, obj_histo, &norm_bck_histo, &norm_obj_histo,
                    &sum_bck_histo, &sum_obj_histo);

    //Can be parallelize easily
    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto idx = y * this->_width + x;
            auto hist_val = grays[idx] / HISTO_FACTOR;

            // Initialize external capacities if not on seeds
            if (this->_excess_flow[idx] == 0.f) {
                float weight_snk = 1.f - std::exp(-norm_bck_histo[hist_val]);
                float weight_src = 1.f - std::exp(-norm_obj_histo[hist_val]);
                this->_excess_flow[idx] = weight_src - weight_snk;
            }
            if (this->_excess_flow[idx] > 0.f)
                this->_nb_active += 1;

            // Initialize neighbors capacities
            initialize_node_capacities(x, y, grays);
        }
    }

    stbi_image_free(labels);
    stbi_image_free(data);

}


void Graph::normalize_histo(size_t *bck_histo, size_t *obj_histo, float **norm_bck_histo,
    float **norm_obj_histo, float *sum_bck_histo, float *sum_obj_histo) {
    size_t bck_max = 0;
    size_t obj_max = 0;

    // Get max for normalization
    for (auto i = 0; i < 256/HISTO_FACTOR; i++) {
        bck_max = bck_histo[i] > bck_max ? bck_histo[i] : bck_max;
        obj_max = obj_histo[i] > obj_max ? obj_histo[i] : obj_max;

    }


    auto tmp_bck_histo = new float[256/HISTO_FACTOR]();
    auto tmp_obj_histo = new float[256/HISTO_FACTOR]();

    // Normalize
    for (auto i = 0; i < 256/HISTO_FACTOR; i++) {
        tmp_bck_histo[i] = bck_histo[i] / float(bck_max);
        tmp_obj_histo[i] = obj_histo[i] / float(obj_max);

        *sum_bck_histo += tmp_bck_histo[i];
        *sum_obj_histo += tmp_obj_histo[i];
    }

    *norm_bck_histo = tmp_bck_histo;
    *norm_obj_histo = tmp_obj_histo;
    return;
}

void Graph::initialize_node_capacities(int x, int y, std::vector<int> &grays) {
    auto idx_curr = y * this->_width + x;
    for (auto i = 0; i < 4; i++) {
        auto idx_nghb = (y + this->y_nghb[i]) * _width + (x + this->x_nghb[i]);
        if (y + y_nghb[i] >= 0 && y + y_nghb[i] < _height &&
            x + x_nghb[i] >= 0 && x + x_nghb[i] < _width)
            this->_neighbors[i][idx_curr] = DISTANCE(grays[idx_curr], grays[idx_nghb]);
    }
}

void Graph::max_flow()
{
    while (any_active()) {
        for (auto y = 0; y < this->_height; y++)
            for (auto x = 0; x < this->_width; x++)
                relabel(x, y);
        for (auto y = 0; y < this->_height; y++)
            for (auto x = 0; x < this->_width; x++)
                push(x, y);
    }
    auto nb_pos = 0;
    auto nb_neg = 0;
    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            if (_excess_flow[y*_width+x] > 0.f)
                nb_pos++;
            else
                nb_neg++;
        }
    }
    std::cout << "Active node: "<<nb_pos<<std::endl;
    std::cout << "Non-Active node: "<<nb_neg<<std::endl;
    std::cout << "Total node: "<<_size<<std::endl;
}

void Graph::push(int x, int y) {
    if (!is_active(x, y))
        return;
    auto idx_curr = y * this->_width + x;

    for (auto i = 0; i < 4; i++) {
        auto idx_nghb = (y + this->y_nghb[i]) * _width + (x + this->x_nghb[i]);
        if (y + y_nghb[i] < 0 || y + y_nghb[i] >= _height ||
            x + x_nghb[i] < 0 || x + x_nghb[i] >= _width)
            continue;
        if (this->_heights[idx_nghb] != this->_heights[idx_curr] - 1)
            continue;
        // Pushed flow
        float flow = std::min(this->_neighbors[i][idx_curr], this->_excess_flow[idx_curr]);
        // Update excess flow
        this->_excess_flow[idx_curr] -= flow;
        this->_excess_flow[idx_nghb] += flow;
        // Update edge capcities
        this->_neighbors[i][idx_curr] -= flow;
        this->_neighbors[id_opp[i]][idx_nghb] += flow;
    }
}

void Graph::relabel(int x, int y) {
    if (!is_active(x, y))
        return;
    auto idx_curr = y * this->_width + x;
    auto tmp_height = HEIGHT_MAX;
    for (auto i = 0; i < 4; i++) {
        auto idx_nghb = (y + this->y_nghb[i]) * _width + (x + this->x_nghb[i]);
        if (this->_neighbors[i][idx_curr] > 0.f)
            tmp_height = std::min(tmp_height, this->_heights[idx_nghb] + 1);
    }
    this->_heights[idx_curr] = tmp_height;
}

bool Graph::is_active(int x, int y) {
    auto idx = y * this->_width + x;
    return this->_excess_flow[idx] > 0.f && this->_heights[idx] < HEIGHT_MAX;
}

bool Graph::any_active() {
    for (auto y = 0; y < this->_height; y++)
        for (auto x = 0; x < this->_width; x++)
        if (is_active(x, y))
            return true;
    return false;
}

void Graph::print() {
    std::ofstream file;
    file.open("output.pmm");

    file << "P3\n" << this->_width << " "<< this->_height << "\n255\n";
    for (int y = 0; y < this->_height; y++)
    {
        for (int x = 0; x < this->_width; x++)
        {
            auto flow = this->_excess_flow[y*_width +x];
            if (flow > 0.f)
                file << "255 255 255 ";
            else
                file << "0 0 0 ";
        }
        file << "\n";
    }
}
