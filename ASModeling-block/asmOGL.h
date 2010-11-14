#ifndef __ASM_OGL_H_
#define __ASM_OGL_H_

namespace asmodeling_block
{
  class GLSys
  {
  public:
    static bool Init(void);
    static void Release(void);
  private:
    static bool _inited;
  };

} // namespace asmodeling_block

#endif //__ASM_OGL_H_