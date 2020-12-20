/**
 * @file perf.c
 * @brief Measure performance: interval timer and hardware counter wrappers
 **/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include "perf.h"
#include <time.h>
#include <unistd.h>

//avoid header conflict
#define CLOCK_MONOTONIC_RAW 4

/* static variable that holds the initial value of the interval timer */
struct itimerval first_u; /* user time */
struct itimerval first_r; /* real time */
struct timespec first_h; /* hardware time */

/*
 * elapsed user time routines
 */

/* init the timer */
void init_etime(void)
{
    first_u.it_interval.tv_sec = 0;
    first_u.it_interval.tv_usec = 0;
    first_u.it_value.tv_sec = MAX_ETIME;
    first_u.it_value.tv_usec = 0;
    setitimer(ITIMER_VIRTUAL, &first_u, NULL);
}

/* return elapsed seconds since call to init_etime */
long double get_etime(void) {
    struct itimerval curr;

    getitimer(ITIMER_VIRTUAL, &curr);
    return (long double) ((first_u.it_value.tv_sec - curr.it_value.tv_sec) +
         (first_u.it_value.tv_usec - curr.it_value.tv_usec)*1e-6);
}

/*
 * elapsed real time routines
 */

/* init the timer */
void init_etime_real(void)
{
    first_r.it_interval.tv_sec = 0;
    first_r.it_interval.tv_usec = 0;
    first_r.it_value.tv_sec = MAX_ETIME;
    first_r.it_value.tv_usec = 0;
    setitimer(ITIMER_REAL, &first_r, NULL);
}

/* return elapsed seconds since call to init_etime_real */
long double get_etime_real(void) {
    struct itimerval curr;

    getitimer(ITIMER_REAL, &curr);
    return (long double) ((first_r.it_value.tv_sec - curr.it_value.tv_sec) +
		     (first_r.it_value.tv_usec - curr.it_value.tv_usec)*1e-6);
}

/*
 * Hardware clock routines
 */
void init_etime_hw(void)
{
    clock_gettime(CLOCK_MONOTONIC_RAW, &first_h);
}

/* return elapsed seconds since call to init_etime_real */
long double get_etime_hw(void)
{
    struct timespec curr;

    clock_gettime(CLOCK_MONOTONIC_RAW, &curr);

    return (long double) ((curr.tv_sec - first_h.tv_sec) +
        (curr.tv_nsec - first_h.tv_nsec)*1e-9);
}
