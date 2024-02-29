#ifndef _STITCHINGENGINE_DEF_
#define _STITCHINGENGINE_DEF_

#ifdef STITCH_EXPORTS
#define STITCH_API extern "C" __declspec(dllexport) 
#else
#define STITCH_API extern "C" __declspec(dllimport)
#endif

#include <utility>
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <fstream>

////////////////////////////////////////////////////////////////////////
enum stError
{
    ESTNoError,                 // ошибок нет
    ESTBadInputParams,          // ошибка во входных параметрах
    ESTCalibrationError,        // ошибка в процессе калибровки
    ESTCamReconstructionError,  // ошибка в процессе реконструкции внутренних параметров камеры
    ESTGetPointsError           // ошибка при поиске соответсвующих точек
};

////////////////////////////////////////////////////////////////////////
enum stWarpType
{
    stWarpPlane,
    stWarpCylindrical,
    stWarpSpherical,
    stWarpFisheye,
    stWarpStereographic,
    srWarpCompressedPlaneA2B1,
    srWarpCompressedPlaneA15B1,
    srWarpCompressedPlanePortraitA2B1,
    srWarpCompressedPlanePortraitA15B1,
    srWarpPaniniA2B1,
    srWarpPaniniA15B1,
    srWarpPaniniPortraitA2B1,
    srWarpPaniniPortraitA15B1,
    srWarpMercator,
    srWarpTransverseMercator
};

////////////////////////////////////////////////////////////////////////
// Функция построения матрицы панорамы.
// На вход подается n-кадров с камер с известными размерами и stride-ом
// следующими по горизонтали в одном порядке (либо слева направо, либо справа налево)
// На выходе получаем матрицы параметров камеры и поворота, размер каждой 3х3
// nFrames - число сшиваемых кадров
// direction - направление сшивки: 0 - горизонтальное, 1 - вертикальное
// widths, heights, strides - ширины, высоты и страйды передаваемых кадров, соответсвенно
// colorSchemes - цветовые схема, сейчас поддерживается серая=8 (8UC1) и 16 битный RGB=16(8UC3)
// frameDataBGR - сами кадры, упакованные в BGR порядке
// outputIntinsicCameraMatrix - результирующие матрицы внутренних параметров камеры
// outputRotationMatrix - результирующие матрицы поворотов
STITCH_API void* createStitching(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                                 const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, stError* outputError);


////////////////////////////////////////////////////////////////////////
// Функция возвращает приобразование исходных точек в панорманые у ранее созданного движка панорамы
// stObject - ранее созданный с помощью createStitching объект панорамы
// stWarpType - тип warping-a
// inputIntinsicCameraMatrix - матрица внутренних параметров камеры
// inputRotationMatrix - матрица поворота 
// nPoints - число точек с иходного изображения
// inputPoints - точки с исходного изображения в формате x1,y1,x2,y2,..,xn,yn, всего 2*nPoints точек
// outputPoints - соответствующие точки на панораме
// nFrame - номер картинки, для которой следует брать матрицы, если это значние равно -1, то используются переданные 
// inputIntinsicCameraMatrix и inputRotationMatrix
// ratio - этот параметр нужен для вызова функции с stObject = 0. 
//         если stObject ненулевой, то этот параметр не используется
STITCH_API void getStitchingPoints(void* stObject, const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                                   const int nPoints, const float* inputPoints, float* outputPoints, stError* outputError, const float ratio, const int nFrame = -1);


////////////////////////////////////////////////////////////////////////
// Удаление движка
STITCH_API void destroyStitching(void* stObject);

#endif //_STITCHINGENGINE_DEF_