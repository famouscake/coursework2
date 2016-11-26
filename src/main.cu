
#include <stdio.h>
#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }
#define THRESH 10000

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


  /*for(int i = 1 ; i < filtered.width - 1 ; i++)*/
  /*{*/
      /*for(int j = 1 ; j < filtered.height - 1 ; j++)*/
      /*{*/
          /*int gradX = getPixel(&grayScale, i-1, j+1) - getPixel(&grayScale, i-1, j-1) + 2*getPixel(&grayScale, i, j+1) - 2*getPixel(&grayScale, i, j-1) + getPixel(&grayScale, i+1, j+1) - getPixel(&grayScale, i+1, j-1);*/

          /*int gradY = getPixel(&grayScale, i-1, j-1) + 2*getPixel(&grayScale, i-1, j) + getPixel(&grayScale, i-1, j+1) - getPixel(&grayScale, i+1, j-1) - 2*getPixel(&grayScale, i+1, j) - getPixel(&grayScale, i+1, j+1);*/

          /*int magnitude = gradX*gradX + gradY*gradY;*/

          /*if (magnitude  > 100000)*/
          /*{*/
              /*setPixel(&filtered, i, j, 255);*/
          /*}*/
          /*else*/
          /*{*/
              /*setPixel(&filtered, i, j, 0);*/
          /*}*/
      /*}*/
  /*}*/

  for(int i = 1 ; i < filtered.height - 1 ; i++)
  {
      for(int j = 1 ; j < filtered.width - 1 ; j++)
      {
          int gradX = getPixel(&grayScale, i-1, j+1) - getPixel(&grayScale, i-1, j-1) + 2*getPixel(&grayScale, i, j+1) - 2*getPixel(&grayScale, i, j-1) + getPixel(&grayScale, i+1, j+1) - getPixel(&grayScale, i+1, j-1);

          int gradY = getPixel(&grayScale, i-1, j-1) + 2*getPixel(&grayScale, i-1, j) + getPixel(&grayScale, i-1, j+1) - getPixel(&grayScale, i+1, j-1) - 2*getPixel(&grayScale, i+1, j) - getPixel(&grayScale, i+1, j+1);

          int magnitude = gradX*gradX + gradY*gradY;

          if (magnitude  > 10000)
          {
              setPixel(&filtered, i, j, 255);
          }
          else
          {
              setPixel(&filtered, i, j, 0);
          }
      }
  }



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

  exit(0);
}
