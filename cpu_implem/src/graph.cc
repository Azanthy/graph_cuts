//
// Created by gigi on 5/22/20.
//

#include "graph.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define HISTO_FACTOR 4

Graph::Graph(char *img, char *seeds) {
    // Load image
    int n, tmp_w, tmp_h;
    unsigned char *data = stbi_load(img, &this->width_, &this->height_, &n, 0);
    //TODO path here is for seeds image, to change for working
    unsigned char *labels = stbi_load(seeds, &tmp_w, &tmp_h, &n, 0);

    this->nodes = std::vector<Node>();
    size_t *bck_histo = new size_t[256/HISTO_FACTOR]();
    size_t *obj_histo = new size_t[256/HISTO_FACTOR]();

    for (auto y = 0; y < this->height_; y++) {
        for (auto x = 0; x < this->width_; x++) {
            auto idx = y * this->width_ * n + x * n;
            char gray = 0.299 * data[idx] + 0.587 * data[idx+1] + 0.114 * data[idx+2];

            // Histograms of size 256/4=64
            bck_histo[gray/HISTO_FACTOR] += labels[idx]   ? 1 : 0;
            obj_histo[gray/HISTO_FACTOR] += labels[idx+2] ? 1 : 0;

            auto node = Node();
            this->nodes.emplace_back();
        }
    }

    float *norm_bck_histo, *norm_obj_histo;
    normalize_histo(bck_histo, obj_histo, &norm_bck_histo, &norm_obj_histo);

    stbi_image_free(labels);
    stbi_image_free(data);

}

void Graph::normalize_histo(size_t *bck_histo, size_t *obj_histo, float **norm_bck_histo, float **norm_obj_histo) {
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
    }

    *norm_bck_histo = tmp_bck_histo;
    *norm_obj_histo = tmp_obj_histo;
    return;
}
