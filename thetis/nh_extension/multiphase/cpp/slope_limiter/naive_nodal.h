// Copyright (C) 2016-2019 Tormod Landet
// SPDX-License-Identifier: Apache-2.0

#ifndef __SLOPE_LIMITER_BASIC_H
#define __SLOPE_LIMITER_BASIC_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/fem/GenericDofMap.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>


namespace dolfin
{


using IntVecIn = Eigen::Ref<const Eigen::VectorXi>;
using DoubleVec = Eigen::Ref<Eigen::VectorXd>;

void naive_nodal_slope_limiter_dg1(const Array<int>& num_neighbours,
                                   const int num_cells_all,
                                   const int num_cells_owned,
                                   const int num_cell_dofs,
                                   const int max_neighbours,
                                   IntVecIn neighbours,
                                   IntVecIn cell_dofs,
                                   IntVecIn cell_dofs_dg0,
                                   DoubleVec exceedances,
                                   DoubleVec results)
{
  if (num_cell_dofs != 3)
    dolfin_error("naive_nodal.h", "naive nodal slope limiter",
                 "C++ NaiveNodal slope limiter only supports DG1");

  // Calculate cell averages (which we will make sure to keep unchanged)
  std::vector<double> averages(num_cells_all);
  for (int ic = 0; ic < num_cells_all; ic++)
  {
    double avg = 0.0;
    for (int id = 0; id < 3; id++)
    {
      avg += results[cell_dofs[ic*3 + id]];
    }
    averages[ic] = avg/3.0;
  }

  // Modify dof values
  for (int ic = 0; ic < num_cells_owned; ic++)
  {
    double avg = averages[ic];
    double exceedance = 0.0;
    double vals[3];

    // Check each dof in this cell
    for (int id = 0; id < 3; id++)
    {
      int dof = cell_dofs[ic*3 + id];
      double dof_val = results[dof];

      // Find highest and lowest value in the connected neighbour cells
      double lo = avg;
      double hi = avg;
      for (int inb = 0; inb < num_neighbours[dof]; inb++)
      {
        int nb = neighbours[dof*max_neighbours + inb];
        double nb_avg = averages[nb];
        lo = std::min(lo, nb_avg);
        hi = std::max(hi, nb_avg);
      }

      // Treat out of bounds values
      if (dof_val < lo)
      {
        vals[id] = lo;
        double ex = dof_val - lo;
        if (std::abs(ex) > std::abs(exceedance)) exceedance = ex;
      }
      else if (dof_val > hi)
      {
        vals[id] = hi;
        double ex = dof_val - hi;
        if (std::abs(ex) > std::abs(exceedance)) exceedance = ex;
      }
      else
      {
        vals[id] = dof_val;
      }
    }

    // Store the maximum absolute cell exceedance and quit early if possible
    exceedances[cell_dofs_dg0[ic]] = exceedance;
    if (exceedance == 0.0) continue;

    // Find the new average and which vertices can be adjusted to obtain the correct average
    double new_avg = (vals[0] + vals[1] + vals[2])/3.0;
    double eps = 0;
    bool moddable[3] = {false, false, false};
    if (std::abs(avg - new_avg) > 1e-15)
    {
     for(int id = 0; id < 3; id++)
     {
       moddable[id] = (new_avg > avg && vals[id] > avg) || (new_avg < avg and vals[id] < avg);
     }
     // Get number of vertex values that can be modified
     int nmod = ((int) moddable[0]) + ((int) moddable[1]) + ((int) moddable[2]);
     if (nmod == 0) {
       dolfin_assert(std::abs(exceedance) < 1e-14);
     }
     else
     {
       eps = (avg - new_avg)*3/nmod;
     }
    }

    // Update the result array
    for(int id = 0; id < 3; id++)
    {
      results[cell_dofs[ic*3 + id]] = vals[id] + eps*((int)moddable[id]);
    }
  }
}


PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("naive_nodal_slope_limiter_dg1", &naive_nodal_slope_limiter_dg1);
}


}

#endif
