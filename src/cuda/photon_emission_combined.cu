

#include <optix.h>
#include <sutil/vec_math.h>
#include "photon_emission/photon_launch_params.h"


extern "C" __constant__ PhotonLaunchParams params;



#include "photon_emission/raygen_photons.cu"
#include "photon_emission/closesthit_store.cu"
#include "photon_emission/miss_photons.cu"
#include "photon_emission/intersection_photon_sphere.cu"



