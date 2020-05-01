#ifndef CONFIG_HH
#define CONFIG_HH

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define PIXEL 8
#define WINDOW 8
#define WINDOW_X 8
#define WINDOW_Y 8

#define SERIAL 0
#if !SERIAL
#define OMP
#endif

#define NUM_THREADS 8
#define NUM_CHANNELS 3
#define TIMER

#endif