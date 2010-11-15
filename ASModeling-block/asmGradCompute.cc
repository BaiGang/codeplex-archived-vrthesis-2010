#include "asmWorkspace.h"

Workspace * g_workspace;

void GradCompute(const ap::real_1d_array& x, ap::real_t& f, ap::real_1d_array& g)
{
	// for each block, set tex img
	
	// set device_g[] to zero
	
	// for each view
	//
	//  g_workspace->straight_render
	//  calc_f
	//
	//  for each perturbing group (block_layer, slice, pu, pv)
	//  	g_workspace->perturb_render
	//  	calc_g

	// copy device_g back to g
}