#ifndef __ASM_TYPES_H_
#define __ASM_TYPES_H_

namespace asmodeling_block
{

enum Orientation
{
  ALONG_X = 0,
  ALONG_Y,
  ALONG_Z
};

typedef int           int32_t;
typedef unsigned int  uint32_t;
typedef float         real_t;
typedef char          int8_t;
typedef unsigned char uint8_t;
typedef short         int16_t;

} // namespace asmodeling_block
#endif //__ASM_TYPES_H_