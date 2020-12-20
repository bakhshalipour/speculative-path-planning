/**
 * @file perf.h
 * @brief This file defines the prototypes for various timer functions
 **/

#ifndef PERF_H
#define PERF_H

#define MAX_ETIME 86400

void init_etime(void);
long double get_etime(void);

void init_etime_real(void);
long double get_etime_real(void);

void init_etime_hw(void);
long double get_etime_hw(void);

#endif /* PERF_H */
