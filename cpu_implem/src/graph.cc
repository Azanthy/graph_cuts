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
#define MANHATTAN(data, i, j)   std::abs(data[i]-data[j]) + std::abs(data[i+1]-data[j+1]) + std::abs(data[i+2]-data[j+2])
#define DISTANCE(data, i, j)    std::exp(-MANHATTAN(data, i, j)/255.f) //TODO grayscale -> rgb
#define BIN_VAL(hist, data, i)  (hist[0][data[i]/4] + hist[1][data[i+1]/4] + hist[2][data[i+2]/4]) / 3.f

Graph::Graph(char *img, char *seeds) {
    // Load image
    int n, tmp_w, tmp_h;
    unsigned char *data = stbi_load(img, &this->_width, &this->_height, &n, 0);
    unsigned char *labels = stbi_load(seeds, &tmp_w, &tmp_h, &n, 0);

    size_t **bck_histo = new size_t*[3]();
    size_t **obj_histo = new size_t*[3]();
    float **norm_bck_histo = new float*[3]();
    float **norm_obj_histo = new float*[3]();
    for (auto i = 0; i < 3; i++) {
        bck_histo[i] = new size_t[256/HISTO_FACTOR]();
        obj_histo[i] = new size_t[256/HISTO_FACTOR]();
        norm_bck_histo[i] = new float[256/HISTO_FACTOR]();
        norm_obj_histo[i] = new float[256/HISTO_FACTOR]();
    }

    this->_img = data;
    this->_size = this->_width * this->_height;
    this->_heights = std::vector<int>(this->_size, 0);
    this->_excess_flow = std::vector<float>(this->_size,0.f);
    for (auto i=0; i < 4; i++)
        this->_neighbors[i] = std::vector<float>(this->_size,0.f);

    // Initialize nodes and histograms
    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto idx = y * this->_width + x;
            auto n_idx = y * this->_width * n + x * n;
            for (auto i = 0; i < 3; i++) {
                // Histograms of size 256/4=64
                if (labels[n_idx] > 220) { // Red color for background
                    bck_histo[i][data[n_idx+i]/HISTO_FACTOR] += 1;
                    this->_excess_flow[idx] = -1.f;
                }
                if (labels[n_idx+2] > 220) { // Blue color for object
                    obj_histo[i][data[n_idx+i]/HISTO_FACTOR] += 1;
                    this->_excess_flow[idx] = 1.f;
                    this->_heights[idx] = HEIGHT_MAX;
                }
            }
        }
    }

    // Compute histograms for external capacities
    float sum_bck_histo = 0;
    float sum_obj_histo = 0;
    normalize_histo(bck_histo, obj_histo, norm_bck_histo, norm_obj_histo,
                    &sum_bck_histo, &sum_obj_histo);

    //Can be parallelize easily
    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto idx = y * this->_width + x;
            auto n_idx = y * this->_width * n + x * n;

            // Initialize external capacities if not on seeds
            if (this->_excess_flow[idx] == 0.f) {
                float weight_snk = 1.f - std::exp(-BIN_VAL(norm_bck_histo, data, n_idx));
                float weight_src = 1.f - std::exp(-BIN_VAL(norm_obj_histo, data, n_idx));
                this->_excess_flow[idx] = weight_src - weight_snk;
            }

            // Initialize neighbors capacities
            initialize_node_capacities(x, y);
        }
    }

    for (auto i=0; i<3;i++){
        delete[] bck_histo[i];
        delete[] obj_histo[i];
        delete[] norm_bck_histo[i];
        delete[] norm_obj_histo[i];
    }
    delete[] bck_histo;
    delete[] obj_histo;
    delete[] norm_bck_histo;
    delete[] norm_obj_histo;
    stbi_image_free(labels);
}

Graph::~Graph()
{
    stbi_image_free(this->_img);
}

void Graph::normalize_histo(size_t **bck_histo, size_t **obj_histo, float **norm_bck_histo,
    float **norm_obj_histo, float *sum_bck_histo, float *sum_obj_histo) {
    size_t bck_max[3] = {0, 0, 0};
    size_t obj_max[3] = {0, 0, 0};

    // Get max for normalization
    for (auto i = 0; i < 256/HISTO_FACTOR; i++) {
        for (auto j = 0; j < 3; j++) {
            bck_max[j] = bck_histo[j][i] > bck_max[j] ? bck_histo[j][i] : bck_max[j];
            obj_max[j] = obj_histo[j][i] > obj_max[j] ? obj_histo[j][i] : obj_max[j];
        }
    }

    // Normalize
    for (auto i = 0; i < 256/HISTO_FACTOR; i++) {
        for (auto j = 0; j < 3; j++) {
            norm_bck_histo[j][i] = bck_histo[j][i] / float(bck_max[j]);
            norm_obj_histo[j][i] = obj_histo[j][i] / float(obj_max[j]);

            *sum_bck_histo += norm_bck_histo[j][i];
            *sum_obj_histo += norm_obj_histo[j][i];
        }
    }

    return;
}

void Graph::initialize_node_capacities(int x, int y) {
    auto idx_curr = y * this->_width + x;
    for (auto i = 0; i < 4; i++) {
        auto idx_nghb = (y + this->y_nghb[i]) * _width + (x + this->x_nghb[i]);
        if (y + y_nghb[i] >= 0 && y + y_nghb[i] < _height &&
            x + x_nghb[i] >= 0 && x + x_nghb[i] < _width) {
            auto norm =std::abs(this->_img[idx_curr] - this->_img[idx_nghb]) +
                       std::abs(this->_img[idx_curr+1] - this->_img[idx_nghb+1]) +
                       std::abs(this->_img[idx_curr+2] - this->_img[idx_nghb+2]);
            this->_neighbors[i][idx_curr] = std::exp(-norm / 384);
        }
    }
}

void Graph::max_flow()
{
    while (any_active()) {
        auto tmp_heights = std::vector<int>(this->_size, 0);
        for (auto y = 0; y < this->_height; y++)
            for (auto x = 0; x < this->_width; x++)
                relabel(x, y, tmp_heights);
        this->_heights = std::vector<int>(tmp_heights);
        for (auto y = 0; y < this->_height; y++)
            for (auto x = 0; x < this->_width; x++)
                push(x, y);
        auto nb_pos = 0;
        auto nb_neg = 0;
        for (auto y = 0; y < this->_height; y++) {
            for (auto x = 0; x < this->_width; x++) {
                if (is_active(x, y))
                    nb_pos++;
                else
                    nb_neg++;
            }
        }
        std::cout << "Active node: "<<nb_pos<<std::endl;
        std::cout << "Non-Active node: "<<nb_neg<<std::endl;
        std::cout << "Total node: "<<_size<<std::endl<<std::endl;
    }

}

//
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

// Code is correct
void Graph::relabel(int x, int y, std::vector<int> &heights) {
    if (!is_active(x, y))
        return;
    auto idx_curr = y * this->_width + x;
    auto tmp_height = HEIGHT_MAX;
    for (auto i = 0; i < 4; i++) {
        auto idx_nghb = (y + this->y_nghb[i]) * _width + (x + this->x_nghb[i]);
        if (this->_neighbors[i][idx_curr] > 0.f)
            tmp_height = std::min(tmp_height, this->_heights[idx_nghb] + 1);
    }
    heights[idx_curr] = tmp_height;
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
            auto n_idx = y * this->_width * 3 + x * 3;
            auto flow = this->_excess_flow[y*_width +x];
            if (_heights[y*_width+x] == HEIGHT_MAX)
                file << int(_img[n_idx]) << " " << int(_img[n_idx+1]) << " "<< int(_img[n_idx+2]) << " ";
            else
                file << "0 0 0 ";
        }
        file << "\n";
    }
}
