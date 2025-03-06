#ifndef CNS_INDEX_H_
#define CNS_INDEX_H_

#if (AMREX_SPACEDIM == 2)

#define URHO  0
#define UMX   1
#define UMY   2
#define UEDEN 3
#define URHOY_F 4
#define URHOY_O 5
#define URHOY_P 6
#define UEINT 7
#define UTEMP 8
#define SFOIL 9
#define NUM_STATE 10
#define NAUX 1

#define NCONS 9
#define NEQNS 7

#define QRHO   0
#define QU     1
#define QV     2
#define QPRES  3
#define QCS    4
#define QEINT  5
#define QTEMP  6
#define NPRIM  7

#else

#define URHO  0
#define UMX   1
#define UMY   2
#define UMZ   3
#define UEDEN 4
#define URHOY 5
#define UEINT 6
#define UTEMP 7
#define SFOIL 8
#define NUM_STATE 9
#define NAUX 1

#define NCONS 8
#define NEQNS 6

#define QRHO   0
#define QU     1
#define QV     2
#define QW     3
#define QPRES  4
#define QCS    5
#define QEINT  6
#define QTEMP  7
#define NPRIM  8

#endif

#define CETA   0
#define CXI    1
#define CLAM   2
#define NCOEF  3

#define NGROW 8

// MACRO FLAGS FOR NAVIER STOKES CHARACTERISTIC BOUNDARY CONDITIONS

#define NSCBCOUT 1002
#define NSCBCIN	 1001

#define SRHO   	0
#define SU     	1
#define SV     	2

#if AMREX_SPACEDIM==2
#define SPRES  	3
#define NPRIMBC 4
#elif AMREX_SPACEDIM==3
#define SW     	3
#define SPRES  	4
#define NPRIMBC 5
#endif


#endif
