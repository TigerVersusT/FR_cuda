extern void franctionalRetinexCuda(float *l_curr, const float *s,
                                   const float *c, const int row,
                                   const int column);

extern void symetricPad(float *img, const int row, const int column,
                        const int n);

extern void utTestCal8DrivCuda(float *img, float *D_l, float *D_l_8,
                               const float *mask, const int row,
                               const int column, const int n);

extern void utTestPadCuda(float *img, const int row, const int column,
                          const int n);