
#include <stdio.h>
#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }
#define THRESH 10000

#define N 480

#define WIDTH 640
#define HEIGHT 480

struct Image {
    int width;
    int height;
    unsigned char *img;
    unsigned char *dev_img;
};

unsigned char getPixel(Image *image, int x, int y)
{
    int id = x * image->width + y;
    return image->img[id];
}

void setPixel(Image *image, int x, int y, unsigned char value)
{
    int id = x * image->width + y;
    image->img[id] = value;
}


    __device__
unsigned char getPixel2(unsigned char *image, int x, int y, int width)
{
    int id = y * WIDTH + x;
    return image[id];
}

    __device__
void setPixel2(unsigned char *image, int x, int y, unsigned char value, int width)
{
    int id = y * WIDTH + x;
    image[id] = value;
}

    __global__
void filter(unsigned char *grayScale, unsigned char *filtered)
{
    /*int tid = (blockIdx.x * gridDim.y * blockDim.x * blockDim.y) + (blockIdx.y * blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;*/

    /*int i = tid / N;*/
    /*int j = tid % N;*/

    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i == 0 || i == WIDTH - 1 || j == 0 || j == HEIGHT - 1)
    {
        return;
    }

    int gradX = getPixel2(grayScale, i-1, j+1, N) - getPixel2(grayScale, i-1, j-1, N) + 2*getPixel2(grayScale, i, j+1, N) - 2*getPixel2(grayScale, i, j-1, N) + getPixel2(grayScale, i+1, j+1, N) - getPixel2(grayScale, i+1, j-1, N);

    int gradY = getPixel2(grayScale, i-1, j-1, N) + 2*getPixel2(grayScale, i-1, j, N) + getPixel2(grayScale, i-1, j+1, N) - getPixel2(grayScale, i+1, j-1, N) - 2*getPixel2(grayScale, i+1, j, N) - getPixel2(grayScale, i+1, j+1, N);

    int magnitude = gradX*gradX + gradY*gradY;

    if (magnitude  > 10000)
    {
        setPixel2(filtered, i, j, 255, N);
    }
    else
    {
        setPixel2(filtered, i, j, 0, N);
    }
}



int main(int argc, char **argv)
{
    Image source;

    if (argc != 3)
    {
        printf("Usage: exec filename filename\n");
        exit(1);
    }
    char *fname = argv[1];
    char *fname2 = argv[2];
    FILE *src;

    if (!(src = fopen(fname, "rb")))
    {
        printf("Couldn't open file %s for reading.\n", fname);
        exit(1);
    }

    char p,s;
    fscanf(src, "%c%c\n", &p, &s);
    if (p != 'P' || s != '6')
    {
        printf("Not a valid PPM file (%c %c)\n", p, s);
        exit(1);
    }

    fscanf(src, "%d %d\n", &source.width, &source.height);
    int ignored;
    fscanf(src, "%d\n", &ignored);

    int pixels = source.width * source.height;
    source.img = (unsigned char *)malloc(pixels*3);
    if (fread(source.img, sizeof(unsigned char), pixels*3, src) != pixels*3)
    {
        printf("Error reading file.\n");
        exit(1);
    }
    fclose(src);

    Image grayScale;
    grayScale.width = source.width;
    grayScale.height = source.height;
    grayScale.img = (unsigned char *)malloc(pixels);
    for (int i = 0; i < pixels; i++)
    {
        unsigned int r = source.img[i*3];
        unsigned int g = source.img[i*3 + 1];
        unsigned int b = source.img[i*3 + 2];
        grayScale.img[i] = 0.2989*r + 0.5870*g + 0.1140*b;
    }

    Image filtered;
    filtered.width = source.width;
    filtered.height = source.height;
    filtered.img = (unsigned char *)malloc(pixels);

    for (int i = 0; i < pixels; i++)
    {
        filtered.img[i] = 0;
    }

    unsigned char *devGrayScale;
    unsigned char *devFiltered;


    // Initialize Cuda Memory
    CHECK(cudaMalloc(&devGrayScale, WIDTH * HEIGHT * sizeof(unsigned char)));
    CHECK(cudaMalloc(&devFiltered, WIDTH * HEIGHT * sizeof(unsigned char)));

    /*// Copy Cuda Memory*/
    CHECK(cudaMemcpy(devGrayScale, grayScale.img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devFiltered, filtered.img, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice));

    /*// Run the kernel*/
    dim3 dimBlock(WIDTH, HEIGHT);
    dim3 dimGrid(1);
    filter<<<dimBlock, dimGrid>>>(devGrayScale, devFiltered);

    /*// Return the Cuda Memory*/
    CHECK(cudaMemcpy(filtered.img, devFiltered, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost));


    FILE *out;
    if (!(out = fopen(fname2, "wb")))
    {
        printf("Couldn't open file for output.\n");
        exit(1);
    }
    fprintf(out, "P5\n%d %d\n255\n", filtered.width, filtered.height);
    if (fwrite(filtered.img, sizeof(unsigned char), pixels, out) != pixels)
    {
        printf("Error writing file.\n");
        exit(1);
    }
    fclose(out);

    free(grayScale.img);
    free(source.img);
    free(filtered.img);

    cudaFree(devGrayScale);
    cudaFree(devFiltered);

    exit(0);
}
