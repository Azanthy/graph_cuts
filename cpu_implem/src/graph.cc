//
// Created by gigi on 5/22/20.
//

#include <cmath>

#include "graph.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define HISTO_FACTOR 4

Graph::Graph(char *img, char *seeds) {
    // Load image
    int n, tmp_w, tmp_h;
    unsigned char *data = stbi_load(img, &this->_width, &this->_height, &n, 0);
    //TODO path here is for seeds image, to change for working
    unsigned char *labels = stbi_load(seeds, &tmp_w, &tmp_h, &n, 0);

    this->_nodes = std::vector<std::shared_ptr<Node>>();
    size_t *bck_histo = new size_t[256/HISTO_FACTOR]();
    size_t *obj_histo = new size_t[256/HISTO_FACTOR]();

    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto idx = y * this->_width * n + x * n;
            char gray = 0.299 * data[idx] + 0.587 * data[idx+1] + 0.114 * data[idx+2];

            auto node = std::make_shared<Node>(gray);
            // Histograms of size 256/4=64
            if (labels[idx]) {
                bck_histo[gray/HISTO_FACTOR] += 1;
                node->_weight_snk = 1;
                node->_weight_src = 0;
            }
            if (labels[idx+2]) {
                obj_histo[gray/HISTO_FACTOR] += 1;
                node->_weight_snk = 0;
                node->_weight_src = 1;
            }
            this->_nodes.emplace_back(node);
        }
    }

    float *norm_bck_histo, *norm_obj_histo;
    float sum_bck_histo = 0;
    float sum_obj_histo = 0;
    normalize_histo(bck_histo, obj_histo, &norm_bck_histo, &norm_obj_histo,
                    &sum_bck_histo, &sum_obj_histo);

    for (auto y = 0; y < this->_height; y++) {
        for (auto x = 0; x < this->_width; x++) {
            auto node = this->_nodes[y * this->_width + x];
            auto idx = node->_gray_val / HISTO_FACTOR;

            // Initialize external weight
            if (node->_weight_snk == -1 && node->_weight_src == -1) {
                node->_weight_snk = 1 - std::exp(-norm_bck_histo[idx]);
                node->_weight_src = 1 - std::exp(-norm_obj_histo[idx]);
            }
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
