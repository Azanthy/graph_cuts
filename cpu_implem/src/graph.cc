//
// Created by gigi on 5/22/20.
//

#include "graph.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Graph::Graph(char *img, char *seeds) {
    // Load image
    int n, tmp_w, tmp_h;
    unsigned char *data = stbi_load(img, &this->width_, &this->height_, &n, 0);
    //TODO path here is for seeds image, to change for working
    unsigned char *labels = stbi_load(seeds, &tmp_w, &tmp_h, &n, 0);

    this->nodes = std::vector<Node>();

    for (auto y = 0; y < this->height_; y++) {
        for (auto x = 0; x < this->width_; x++) {
            auto idx = y * this->width_ * n + x * n;
            char gray = 0.299 * data[idx] + 0.587 * data[idx+1] + 0.114 * data[idx+2];

            // Histograms of size 256/4=64
            bck_histo[gray/4] += labels[idx];
            obj_histo[gray/4] += labels[idx+2];

            auto node = Node();
            this->nodes.emplace_back();
        }
    }

    stbi_image_free(labels);
    stbi_image_free(data);

}
