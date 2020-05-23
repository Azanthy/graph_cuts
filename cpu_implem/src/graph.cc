//
// Created by gigi on 5/22/20.
//

#include <cmath>

#include "graph.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define HISTO_FACTOR 4
#define DISTANCE(x,y) std::exp(-std::abs(x - y)/255.f)

Graph::Graph(char *img, char *seeds) {
    // Load image
    int n, tmp_w, tmp_h;
    unsigned char *data = stbi_load(img, &this->_width, &this->_height, &n, 0);
    unsigned char *labels = stbi_load(seeds, &tmp_w, &tmp_h, &n, 0);

    this->_nodes = std::vector<std::shared_ptr<Node>>();
    size_t *bck_histo = new size_t[256/HISTO_FACTOR]();
    size_t *obj_histo = new size_t[256/HISTO_FACTOR]();

    // Initialize nodes and histograms
    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto idx = y * this->_width * n + x * n;
            char gray = 0.299 * data[idx] + 0.587 * data[idx+1] + 0.114 * data[idx+2];

            auto node = std::make_shared<Node>(gray);
            // Histograms of size 256/4=64
            if (labels[idx]) {
                bck_histo[gray/HISTO_FACTOR] += 1;
                node->_excess_flow = -1.f;
            }
            if (labels[idx+2]) {
                obj_histo[gray/HISTO_FACTOR] += 1;
                node->_excess_flow = 1.f;
            }
            this->_nodes.push_back(node);
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
            auto node = this->_nodes[y * this->_width + x];
            auto idx = node->_gray_val / HISTO_FACTOR;

            // Initialize external capacities if not on seeds
            if (node->_excess_flow == 0.f) {
                float weight_snk = 1.f - std::exp(-norm_bck_histo[idx]);
                float weight_src = 1.f - std::exp(-norm_obj_histo[idx]);
                node->_excess_flow = weight_src - weight_snk;
            }

            // Initialize neighbors capacities
            initialize_node_capacities(x, y, node);
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

void Graph::initialize_node_capacities(int x, int y, shared_node node) {
    // Up
    if (y > 0) {
        auto up_node = _nodes[(y-1) * _width + x];
        node->_capacities[0] = DISTANCE(node->_gray_val, up_node->_gray_val);
        auto tmp = 1 - std::exp(-std::abs(node->_gray_val - up_node->_gray_val)/255.f);
    }
    // Right
    if (x < this->_width-1) {
        auto right_node = _nodes[y * _width + (x+1)];
        node->_capacities[1] = DISTANCE(node->_gray_val, right_node->_gray_val);
    }
    // Bottom
    if (y < this->_height-1) {
        auto bottom_node = _nodes[(y+1) * _width + x];
        node->_capacities[2] = DISTANCE(node->_gray_val, bottom_node->_gray_val);
    }
    // Left
    if (x > 0) {
        auto left_node = _nodes[y * _width + (x-1)];
        node->_capacities[3] = DISTANCE(node->_gray_val, left_node->_gray_val);
    }
}
