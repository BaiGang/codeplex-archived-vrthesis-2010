// A simple timer
// by baigang

#ifndef _CPU_TIMER_H_
#define _CPU_TIMER_H_

#include <sys/timeb.h>

class Timer
{

public:
  inline Timer() {};
  inline ~Timer() {};

  inline void start(void)
  {
    ftime((struct timeb *)&tb_before);
  }

  inline double stop(void)
  {
    // Get timestamp
    ftime((struct timeb *)&tb_after);

    // Compute elapsed time
    long secs_elapsed = tb_after.time - tb_before.time;
    long msecs_elapsed = tb_after.millitm - tb_before.millitm;
    if (msecs_elapsed < 0)
    {
      secs_elapsed--;
      msecs_elapsed+=1000;
    }

    double ret =  ((double)secs_elapsed) + ((double)msecs_elapsed) * 0.001;

    return ret;
  }

private:
  struct timeb tb_before;
  struct timeb tb_after;
};


#endif //_CPU_TIMER_H_
