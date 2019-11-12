// Copyright (C) 2015-2019 Tormod Landet
// SPDX-License-Identifier: Apache-2.0

#ifndef __GRADIENT_RECONSTRUCTION_H
#define __GRADIENT_RECONSTRUCTION_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/fem/GenericDofMap.h>
#include <Eigen/Core>

namespace dolfin
{

using IntVecIn = Eigen::Ref<const Eigen::VectorXi>;
using IntMatIn = Eigen::Ref<const Eigen::MatrixXi>;
using DoubleVecIn = Eigen::Ref<const Eigen::VectorXd>;

void reconstruct_gradient(const Function& alpha_function,
                          IntVecIn num_neighbours,
                          IntMatIn neighbours,
                          DoubleVecIn lstsq_matrices,
                          DoubleVecIn lstsq_inv_matrices,
                          std::vector<std::shared_ptr<Function>>& gradient)
{
  // Function spaces and dofmaps of alpha and gradient are the same
  const FunctionSpace& V = *alpha_function.function_space();
  const auto dofmap = V.dofmap();

  const Mesh& mesh = *V.mesh();
  const std::size_t ndim = mesh.geometry().dim();
  const std::size_t num_cells_owned = neighbours.rows();
  const std::size_t num_neighbours_max = neighbours.cols();

  // Check dimensions
  if (lstsq_matrices.size() != num_neighbours.size()*ndim*num_neighbours_max)
    throw std::length_error("ERROR: lstsq_matrices.size() != num_neighbours.size()*ndim*num_neighbours_max");
  if (lstsq_inv_matrices.size() != num_neighbours.size()*ndim*ndim)
      throw std::length_error("ERROR: lstsq_matrices.size() != num_neighbours.size()*ndim*ndim");

  // Get ghosted size
  const auto im = dofmap->index_map();
  const auto num_dofs_local_and_ghosts = im->size(dolfin::IndexMap::MapSize::ALL);

  // Local indices of ghosted and normal dofs
  std::vector<la_index> indices(num_dofs_local_and_ghosts);
  for (std::size_t i = 0; i < num_dofs_local_and_ghosts; i++)
    indices[i] = i;

  // Get alpha values with ghost values included at the end of the buffer
  std::vector<double> a_vec(num_dofs_local_and_ghosts);
  alpha_function.vector()->get_local(a_vec.data(), a_vec.size(), indices.data());

  // Create local gradient vectors
  std::vector<std::vector<double>> grad_vec(ndim);
  for (std::size_t d = 0; d < ndim; d++)
    gradient[d]->vector()->get_local(grad_vec[d]);

  double ATdotB[ndim];
  double grad[ndim];
  for (std::size_t icell = 0; icell < num_cells_owned; icell++)
  {
    // Reset ATdotB
    for (std::size_t d = 0; d < ndim; d++)
    {
      ATdotB[d] = 0.0;
    }

    // Get the value in this cell
    const la_index dix = dofmap->cell_dofs(icell)[0];
    const double a0 = a_vec[dix];

    // Compute the transpose(A)*B  matrix vector product
    int start = icell * ndim * num_neighbours_max;
    for (int n = 0; n < num_neighbours[icell]; n++)
    {
      const la_index nidx = neighbours(icell, n);
      const la_index ndix = dofmap->cell_dofs(nidx)[0];
      double aN = a_vec[ndix];
      for (std::size_t d = 0; d < ndim; d++)
      {
        ATdotB[d] += lstsq_matrices[start + d * num_neighbours_max + n] * (aN - a0);
      }
    }

    // Compute the inv(AT*A) * ATdotB matrix vector product
    start = icell * ndim * ndim;
    for (std::size_t d = 0; d < ndim; d++)
    {
      grad[d] = 0.0;
      for (std::size_t d2 = 0; d2 < ndim; d2++)
      {
        grad[d] += lstsq_inv_matrices[start + d * ndim + d2] * ATdotB[d2];
      }
      const la_index dof = dofmap->cell_dofs(icell)[0];
      grad_vec[d][dof] = grad[d];
    }
  }
  for (std::size_t d = 0; d < ndim; d++)
  {
    gradient[d]->vector()->set_local(grad_vec[d]);
    gradient[d]->vector()->apply("insert");
  }
}

}

#endif
