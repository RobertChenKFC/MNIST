/* Copyright 2018 RobertChenKFC */

#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

static const std::size_t X_HEADER_SIZE  = 16;
static const std::size_t Y_HEADER_SIZE  = 8;

struct Photo {
  uint8_t pixels[784];
};

/* Please remember to FREE memblock */
char* GetMemblock(const std::string &fileName) {
  std::ifstream file(fileName,
                     std::ios::binary | std::ios::in | std::ios::ate);
  if (file.is_open()) {
    std::size_t size = file.tellg();

    // Read into memory
    char *memblock = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(memblock, size);

    file.close();

    return memblock;
  } else {
    std::cout << "Unable to open file " << fileName << std::endl;
    return nullptr;
  }
}

void GetPhotos(char *memblock, Photo *photo, std::size_t numPhotos) {
  memcpy(photo, memblock + X_HEADER_SIZE, sizeof(Photo) * numPhotos);
}

void InvertPixels(Photo *photo) {
  for (std::size_t i = 0; i < 784; i++)
    photo->pixels[i] = 255 - photo->pixels[i];
}

void OutputPhotos(const std::string &fileName,
                  Photo *photos, std::size_t numPhotos) {
  std::ofstream file(fileName, std::ios::out);
  if (file.is_open()) {
    for (std::size_t i = 0; i < numPhotos; i++) {
      for (std::size_t j = 0; j < 784; j++)
        file << (photos[i].pixels[j] / 255.f) << ' ';
      file << std::endl;
    }
    file.close();
  } else {
    std::cout << "Unable to open file " << fileName << std::endl;
  }
}

void GetLabels(char *memblock, uint8_t *labels, std::size_t numLabels) {
  memcpy(labels, memblock + Y_HEADER_SIZE, numLabels);
}

void OutputLabels(const std::string &fileName,
                  uint8_t *labels, std::size_t numLabels) {
  std::ofstream file(fileName, std::ios::out);
  if (file.is_open()) {
    for (std::size_t i = 0; i < numLabels; i++) {
      for (std::size_t j = 0; j <= 9; j++) {
        if (j == labels[i])
          file << "1 ";
        else
          file << "0 ";
      }
      file << std::endl;
    }
    file.close();
  } else {
    std::cout << "Unable to open file " << fileName << std::endl;
  }
}

int main() {
  // Get training photos
  char *memblock = GetMemblock("train-images-idx3-ubyte");
  Photo *trainingPhotos = new Photo[60000];
  GetPhotos(memblock, trainingPhotos, 60000);
  std::cout << "Training photos read" << std::endl;
  // Invert training photos
  for (std::size_t i = 0; i < 60000; i++)
    InvertPixels(&trainingPhotos[i]);
  // Output to train_xs.out
  OutputPhotos("train_xs.out", trainingPhotos, 60000);
  std::cout << "Output train_xs.out" << std::endl;
  delete[] trainingPhotos;
  delete[] memblock;

  // Get training labels
  memblock = GetMemblock("train-labels-idx1-ubyte");
  uint8_t trainingLabels[60000];
  GetLabels(memblock, trainingLabels, 60000);
  std::cout << "Training labels read" << std::endl;
  // Output to train_ys.out
  OutputLabels("train_ys.out", trainingLabels, 60000);
  std::cout << "Output train_ys.out" << std::endl;
  delete[] memblock;

  // Get testing photos
  memblock = GetMemblock("t10k-images-idx3-ubyte");
  Photo *testingPhotos = new Photo[10000];
  GetPhotos(memblock, testingPhotos, 10000);
  std::cout << "Testing photos read" << std::endl;
  // Invert testing photos
  for (std::size_t i = 0; i < 10000; i++)
    InvertPixels(&testingPhotos[i]);
  // Output to test_xs.out
  OutputPhotos("test_xs.out", testingPhotos, 10000);
  std::cout << "Output test_xs.out" << std::endl;
  delete[] testingPhotos;
  delete[] memblock;

  // Get testing labels
  memblock = GetMemblock("t10k-labels-idx1-ubyte");
  uint8_t testingLabels[10000];
  GetLabels(memblock, testingLabels, 10000);
  std::cout << "Testing labels read" << std::endl;
  // Output to train_ys.out
  OutputLabels("test_ys.out", testingLabels, 10000);
  std::cout << "Output test_ys.out" << std::endl;
  delete[] memblock;
}
