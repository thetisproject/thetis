// Copyright (C) 2017-2019 Tormod Landet
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <tuple>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>


namespace dolfin {


class LocalMaximaMeasurer
{
public:
  LocalMaximaMeasurer(const Mesh &mesh)
  {
    _mesh = &mesh;
    setup_connectivity();
  }

  /*
   * Measure the amount of overshoots/undershoots caused by local maxima at dof locations
   */
  double measure(Function &u)
  {
    // Get processor local dof values
    const auto dm = u.function_space()->dofmap();
    std::vector<double> dof_values;
    u.vector()->get_local(dof_values);

    // Calculate cell average values for all cells, not only "regular" cells
    const std::size_t Ncells = _mesh->num_cells();
    std::vector<double> cell_averages(Ncells);
    for (std::size_t cid = 0; cid < Ncells; cid++)
    {
      const auto cell_dofs = dm->cell_dofs(cid);
      assert(cell_dofs.size() == 6);
      const std::size_t d0 = cell_dofs[3];
      const std::size_t d1 = cell_dofs[4];
      const std::size_t d2 = cell_dofs[5];
      cell_averages[cid] = (dof_values[d0] + dof_values[d1] + dof_values[d2]) / 3.0;
    }

    // Calculate overshoot
    double overshoot = 0.0;
    const std::size_t Ncells_owned = _mesh->topology().ghost_offset(_ndim);
    for (std::size_t cid = 0; cid < Ncells_owned; cid++)
    {
      const auto& cell_neighbours = _neighbours[cid];
      const auto cell_dofs = dm->cell_dofs(cid);
      const auto num_cell_dofs = cell_dofs.size();
      for (std::size_t idof = 0; idof < num_cell_dofs; idof++)
      {
        const auto dof = cell_dofs[idof];
        const auto& nbs = cell_neighbours[idof];
        if (nbs.size() == 0) continue;

        // Find allowable high and low values
        double lim_low = cell_averages[cid];
        double lim_high = lim_low;
        for (const auto nb : nbs)
        {
          lim_low = std::min(lim_low, cell_averages[nb]);
          lim_high = std::max(lim_high, cell_averages[nb]);
        }

        // Compute overshoot for this dof
        const double value = dof_values[dof];
        double scale = 1.0 / std::max(lim_high - lim_low, 1e-8);
        if (value < lim_low)
        {
          overshoot += (lim_low - value) * scale;
        }
        else if (value > lim_high)
        {
          overshoot += (value - lim_high) * scale;
        }
      }
    }
    // Sum all overshoots across processors
    MPI::sum(MPI_COMM_WORLD, overshoot);
    return overshoot;
  }

private:
  // Data from outside
  const Mesh *_mesh;
  const int _ndim = 2;

  // Neighbour cells for all vertices and facets for each of the cells
  std::vector<std::vector<std::vector<std::size_t> > > _neighbours;

  /*
   * For each cell and for each of the 3+3 vertices/facets in this
   * cell find the ids of all the connected neighbour cells (not
   * including the cell itself
   * */
  void setup_connectivity()
  {
    auto &mesh = *_mesh;
    assert(mesh.topology().dim() == _ndim);
    auto connectivity_CV = mesh.topology()(2, 0);
    auto connectivity_VC = mesh.topology()(0, 2);
    auto connectivity_CF = mesh.topology()(2, 1);
    auto connectivity_FC = mesh.topology()(1, 2);

    for (CellIterator cell(mesh, "regular"); !cell.end(); ++cell)
    {
      const std::size_t cid = cell->index();
      _neighbours.emplace_back();
      std::vector<std::vector<std::size_t> > &cell_neighbours = _neighbours[cid];

      // Gather cells connected to each of the vertices
      const unsigned int* vertices = connectivity_CV(cid);
      int nvert_for_cell = connectivity_CV.size_global(cid);
      for (int iv = 0; iv < nvert_for_cell; iv++)
      {
        const std::size_t vid = vertices[iv];
        cell_neighbours.emplace_back();
        std::vector<std::size_t> &vert_neighbours = cell_neighbours[iv];

        // Look at cells connected to this vertex
        const unsigned int* cells = connectivity_VC(vid);
        int ncells_for_vert = connectivity_VC.size_global(vid);
        for (int ic = 0; ic < connectivity_VC.size_global(vid); ic++)
        {
          const std::size_t nbid = cells[ic];
          if (nbid != cid) vert_neighbours.push_back(nbid);
        }
      }

      // Gather cells connected to each of the facets
      const unsigned int* facets = connectivity_CF(cid);
      int nfacets_for_cell = connectivity_CF.size_global(cid);
      for (int ifa = 0; ifa < nfacets_for_cell; ifa++)
      {
        const std::size_t fid = facets[ifa];
        cell_neighbours.emplace_back();
        std::vector<std::size_t> &facet_neighbours = cell_neighbours[ifa + 3];

        // Look at cells connected to this facet
        const unsigned int* cells = connectivity_FC(fid);
        int ncells_for_facet = connectivity_FC.size_global(fid);
        for (int ic = 0; ic < connectivity_FC.size_global(fid); ic++)
        {
          const std::size_t nbid = cells[ic];
          if (nbid != cid) facet_neighbours.push_back(nbid);
        }
      }
    }
  }
}; // end class


PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<LocalMaximaMeasurer>(m, "LocalMaximaMeasurer")
      .def(pybind11::init<const Mesh &>())
      .def("measure", &LocalMaximaMeasurer::measure);
}

} // end namespace dolfin
