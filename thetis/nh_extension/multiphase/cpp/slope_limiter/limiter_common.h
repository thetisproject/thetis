// Copyright (C) 2017-2019 Tormod Landet
// SPDX-License-Identifier: Apache-2.0

#ifndef __SLOPE_LIMITER_COMMON_H
#define __SLOPE_LIMITER_COMMON_H
#include <cstdint>
#include <limits>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>

namespace dolfin
{

// Row major matrices
using MatIntRM = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatDoubleRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Input reference types
using IntVecIn = Eigen::Ref<const Eigen::VectorXi>;
using IntMatIn = Eigen::Ref<const MatIntRM>;
using DoubleVecIn = Eigen::Ref<const Eigen::VectorXd>;
using DoubleMatIn = Eigen::Ref<const MatDoubleRM>;

enum BoundaryDofType
{
  NOT_ON_BOUNDARY = 0,
  DIRICHLET = 1,
  NEUMANN = 2,
  ROBIN = 3,
  OTHER = 4
};

struct SlopeLimiterInput
{
  // --------------------------------------------------------------------------
  // Connectivity and dofs
  // --------------------------------------------------------------------------

  // Number of cells owned by the local process
  int num_cells_owned;

  // Number of neighbours for each dof (dimension Ndof)
  Eigen::VectorXi num_neighbours;

  // Neighbours for each dof (dimension Ndof,max_neighbours)
  MatIntRM neighbours;

  // Dofs for each cell (dimension num_cells_owned * ndofs per cell)
  MatIntRM cell_dofs;
  Eigen::VectorXi cell_dofs_dg0;

  // Coordinates of the cell vertices
  MatIntRM cell_vertices;
  MatDoubleRM vertex_coords;

  // Coordinate of cell midpoints
  MatDoubleRM cell_midpoints;

  // We can clamp the limited values to a given range
  double global_min = std::numeric_limits<double>::lowest();
  double global_max = std::numeric_limits<double>::max();

  // Treat Robin BCs as Dirichlet in the sence that the Robin Dirichlet
  // value is included in the valid value range for boundary dofs
  bool trust_robin_dval = true;

  void set_arrays(const int num_cells_owned,
                  IntVecIn num_neighbours,
                  IntMatIn neighbours,
                  IntMatIn cell_dofs,
                  IntVecIn cell_dofs_dg0,
                  IntMatIn cell_vertices,
                  DoubleMatIn vertex_coords)
  {
    const int Ndofs_DGX = num_neighbours.size();
    const int Ncells = cell_dofs.rows();
    const int D = vertex_coords.cols();
    const int Nvert_cell = D + 1;
    const int Ndofs = cell_dofs.rows() * cell_dofs.cols();

    // Verify that the input data shapes are correct
    if (num_cells_owned > Ncells)
      throw std::length_error("ERROR: num_cells_owned > Ncells");
    if (neighbours.rows() != Ndofs_DGX)
      throw std::length_error("ERROR: neighbours.rows() != Ndofs_DGX");
    if (cell_dofs_dg0.size() != Ncells)
      throw std::length_error("ERROR: cell_dofs_dg0.size() != Ncells");
    if (cell_vertices.rows() != Ncells)
      throw std::length_error("ERROR: cell_vertices.rows() != Ncells");
    if (cell_vertices.cols() != Nvert_cell)
      throw std::length_error("ERROR: cell_vertices.cols() != Nvert_cell");

    // Verify that vertex indices are correct
    for (int i = 0; i < Ncells; i++)
    {
      for (int j = 0; j < Nvert_cell; j++)
      {
        int v = cell_vertices(i, j);
        if (v < 0 or v >= vertex_coords.rows())
          throw std::length_error("ERROR: v < 0 or v >= vertex_coords.rows()");
      }
    }

    // Verify that dof indices are correct
    for (int i = 0; i < Ncells; i++)
    {
      int d0 = cell_dofs_dg0(i);
      if (d0 < 0 or d0 >= Ncells)
        throw std::length_error("ERROR: d0 < 0 or d0 >= Ncells");

      for (int j = 0; j < cell_dofs.cols(); j++)
      {
        int d = cell_dofs(i, j);
        if (d < 0 or d >= Ndofs)
          throw std::length_error("ERROR: d < 0 or d >= Ndofs");
      }
    }

    // Verify that dof neighbour indices are correct
    for (int i = 0; i < Ndofs_DGX; i++)
    {
      for (int j = 0; j < num_neighbours(i); j++)
      {
        int c = neighbours(i, j);
        if (c < 0 or c >= Ncells)
          throw std::length_error("ERROR: c < 0 or c >= Ncells_owned");
      }
    }

    this->num_cells_owned = num_cells_owned;
    this->num_neighbours = num_neighbours;
    this->neighbours = neighbours;
    this->cell_dofs = cell_dofs;
    this->cell_dofs_dg0 = cell_dofs_dg0;
    this->cell_vertices = cell_vertices;
    this->vertex_coords = vertex_coords;

    // Compute cell midpoints
    cell_midpoints.resize(Ncells, D);
    cell_midpoints.setZero();
    for (int i = 0; i < Ncells; i++)
    {
      for (int j = 0; j < Nvert_cell; j++)
      {
        int v = cell_vertices(i, j);
        for (int k = 0; k < D; k++)
          cell_midpoints(i, k) += vertex_coords(v, k) / (D + 1);
      }
    }
  }

  // Should we limit a given cell. Look up with cell number and get 1 or 0
  Eigen::VectorXi limit_cell;

  void set_limit_cell(IntVecIn limit_cell)
  {
    if (limit_cell.size() != this->num_cells_owned)
      throw std::length_error("ERROR: limit_cell.size() != this->num_cells_owned");
    this->limit_cell = limit_cell;
  }

  // --------------------------------------------------------------------------
  // Boundary conditions
  // --------------------------------------------------------------------------

  std::vector<BoundaryDofType> boundary_dof_type;
  Eigen::VectorXd boundary_dof_value;
  bool enforce_boundary_conditions;

  void set_boundary_values(IntVecIn boundary_dof_type,
                           DoubleVecIn boundary_dof_value,
                           const bool enforce_bcs)
  {
    const int Ndofs = num_cells_owned * cell_dofs.cols();
    if (boundary_dof_type.size() != Ndofs or boundary_dof_value.size() != Ndofs)
      throw std::length_error("ERROR: boundary_dof_type.size() != Ndofs or boundary_dof_value.size() != Ndofs");

    this->boundary_dof_type.resize(boundary_dof_type.size());
    for (int i = 0; i < boundary_dof_type.size(); i++)
      this->boundary_dof_type[i] = static_cast<BoundaryDofType>(boundary_dof_type[i]);

    this->boundary_dof_value = boundary_dof_value;
    this->enforce_boundary_conditions = enforce_bcs;
  }
};

} // end namespace dolfin

#endif
