// Copyright (C) 2017-2019 Tormod Landet
// SPDX-License-Identifier: Apache-2.0

#include <dolfin/mesh/Facet.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/fem/GenericDofMap.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace dolfin
{

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DoubleVec = Eigen::Ref<Eigen::VectorXd>;
using IntVecIn = Eigen::Ref<const Eigen::VectorXi>;
using DoubleVecIn = Eigen::Ref<const Eigen::VectorXd>;
using DoubleMatIn = Eigen::Ref<const RowMatrixXd>;


class ConvectionBlendingInput {
public:
  ConvectionBlendingInput() {};

  // Dofmaps
  Eigen::VectorXi cell_dofmap;
  Eigen::VectorXi facet_dofmap;
  void set_dofmap(IntVecIn cell_dofmap, IntVecIn facet_dofmap)
  {
    this->cell_dofmap = cell_dofmap;
    this->facet_dofmap = facet_dofmap;
  }

  // Facet info
  Eigen::VectorXd facet_area;
  RowMatrixXd facet_normal;
  RowMatrixXd facet_midpoint;
  void set_facet_info(DoubleVecIn areas, DoubleMatIn normals, DoubleMatIn midpoints)
  {
    facet_area = areas;
    facet_normal = normals;
    facet_midpoint = midpoints;
  }

  // Cell info
  Eigen::VectorXd cell_volume;
  RowMatrixXd cell_midpoint;
  void set_cell_info(DoubleVecIn volumes, DoubleMatIn midpoints)
  {
    cell_volume = volumes;
    cell_midpoint = midpoints;
  }

};


template <const std::size_t ndim>
double hric(const ConvectionBlendingInput& inp,
            const Mesh& mesh,
            const DoubleVecIn& a_vec,
            const DoubleMatIn& g_vec,
            const DoubleMatIn& v_vec,
            DoubleVec& b_vec,
            const double dt,
            const std::string variant)
{
  typedef Eigen::Matrix<double, ndim, 1> Vec;

  auto conFC = mesh.topology()(ndim - 1, ndim);
  const double EPS = 1.0e-6;
  double Co_max = 0.0;
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    auto fidx = facet->index();
    auto fdof = inp.facet_dofmap[fidx];

    // Skip exterior cells (which do not have two connected cells)
    if (conFC.size(fidx) != 2)
    {
      b_vec[fdof] = 0.0;
      continue;
    }

    // Indices of the two local cells
    const unsigned int* tmp = conFC(fidx);
    const unsigned int ic0(*tmp), ic1(*(++tmp));

    // Midpoint of local cells
    Vec cell0_mp = inp.cell_midpoint.row(ic0);
    Vec cell1_mp = inp.cell_midpoint.row(ic1);
    Vec mp_dist = cell1_mp - cell0_mp;

    // Velocity at the facet
    Vec ump;
    for (std::size_t d = 0; d < ndim; d++)
      ump[d] = v_vec(d, fdof);

    // Normal on facet pointing out of cell 0
    Vec normal = inp.facet_normal.row(fidx);

    // Find indices of downstream ("D") cell and central ("C") cell
    double uf = normal.dot(ump);
    unsigned int iaC(ic0), iaD(ic1);
    Vec vec_to_downstream = mp_dist;
    if (uf <= 0)
    {
      iaC = ic1;
      iaD = ic0;
      vec_to_downstream *= -1.0;
    }

    // Find alpha in D and C cells
    int dofD = inp.cell_dofmap[iaD];
    int dofC = inp.cell_dofmap[iaC];
    double aD = a_vec[dofD];
    double aC = a_vec[dofC];

    if (std::abs(aC - aD) < EPS)
    {
      // No change in this area, use upstream value
      b_vec[fdof] = 0.0;
      continue;
    }

    // Gradient of alpha in the central cell
    Vec gC;
    for (std::size_t d = 0; d < ndim; d++)
      gC[d] = g_vec(d, dofC);
    double gC_sq = gC.dot(gC);

    if (gC_sq == 0)
    {
      // No change in this area, use upstream value
      b_vec[fdof] = 0.0;
      continue;
    }

    // Upstream value
    // See Ubbink's PhD (1997) equations 4.21 and 4.22
    double aU = aD - 2 * gC.dot(vec_to_downstream);
    aU = std::min(std::max(aU, 0.0), 1.0);

    // Calculate the facet Courant number
    double Co = std::abs(uf) * dt * inp.facet_area[fidx] / inp.cell_volume[iaC];
    Co_max = std::max(Co_max, Co);

    if (std::abs(aU - aD) < EPS)
    {
      // No change in this area, use upstream value
      b_vec[fdof] = 0.0;
      continue;
    }

    // Angle between face normal and surface normal
    double n_sq = normal.dot(normal);
    double cos_theta = normal.dot(gC) / std::pow(n_sq * gC_sq, 0.5);

    // Introduce normalized variables
    double tilde_aC = (aC - aU) / (aD - aU);
    double tilde_aF_final;

    if (tilde_aC <= 0 or tilde_aC >= 1)
    {
      // Only upwind is stable
      b_vec[fdof] = 0.0;
      continue;
    }

    if (variant == "HRIC")
    {
      // Compressive scheme
      double tilde_aF = 1;
      if (0 <= tilde_aC and tilde_aC <= 0.5) tilde_aF = 2 * tilde_aC;

      // Correct tilde_aF to avoid aligning with interfaces
      double t = std::pow(std::abs(cos_theta), 0.5);
      double tilde_aF_star = tilde_aF * t + tilde_aC * (1 - t);

      // Correct tilde_af_star for high Courant numbers
      if (Co < 0.4)
        tilde_aF_final = tilde_aF_star;
      else if (Co < 0.75)
        tilde_aF_final = tilde_aC + (tilde_aF_star - tilde_aC) * (0.75 - Co) / (0.75 - 0.4);
      else
        tilde_aF_final = tilde_aC;
    }
    else {
      throw std::invalid_argument("HRIC variant " + variant + " not supported by C++ impl.");
    }

    // Avoid tilde_aF being slightly lower that tilde_aC due to
    // floating point errors, it must be greater or equal
    if (tilde_aC - EPS < tilde_aF_final and tilde_aF_final < tilde_aC)
      tilde_aF_final = tilde_aC;

    // Calculate the downstream blending factor (0=upstream, 1=downstream)
    b_vec[fdof] = (tilde_aF_final - tilde_aC) / (1 - tilde_aC);

    if (b_vec[fdof] < 0.0 or b_vec[fdof] > 1.0)
      throw std::domain_error("HRIC ERROR: blending factor is out of range. This should never happen!");
  }
  return Co_max;
}


PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<ConvectionBlendingInput>(m, "ConvectionBlendingInput")
    .def(pybind11::init())
    .def("set_dofmap", &ConvectionBlendingInput::set_dofmap)
    .def("set_facet_info", &ConvectionBlendingInput::set_facet_info)
    .def("set_cell_info", &ConvectionBlendingInput::set_cell_info)
    .def_readwrite("cell_dofmap", &ConvectionBlendingInput::cell_dofmap)
    .def_readwrite("facet_dofmap", &ConvectionBlendingInput::facet_dofmap);
  m.def("hric_2D", &hric<2>);
  m.def("hric_3D", &hric<3>);
  m.def("reconstruct_gradient", &reconstruct_gradient);
}

} // end namespace dolfin
