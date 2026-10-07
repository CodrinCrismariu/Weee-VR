// Hand-written stand-in for the header Ceres' CMake generates from cmake/config.h.in.
// Matches Ceres' own iOS configuration: Eigen sparse backend, no SuiteSparse/CXSparse/LAPACK,
// C++11 threads, miniglog. Only the fully dynamic Schur eliminator is compiled.
#ifndef CERES_PUBLIC_INTERNAL_CONFIG_H_
#define CERES_PUBLIC_INTERNAL_CONFIG_H_

#define CERES_USE_EIGEN_SPARSE
#define CERES_NO_LAPACK
#define CERES_NO_SUITESPARSE
#define CERES_NO_CXSPARSE
#define CERES_RESTRICT_SCHUR_SPECIALIZATION
#define CERES_USE_CXX11_THREADS

#endif  // CERES_PUBLIC_INTERNAL_CONFIG_H_
