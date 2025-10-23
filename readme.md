# 🌊 Thetis

[![Weekly Tests (master)](https://github.com/thetisproject/thetis/actions/workflows/weekly-master.yml/badge.svg?branch=master)](https://github.com/thetisproject/thetis/actions/workflows/weekly-master.yml)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

**Thetis** is an open-source, finite element framework for simulating **coastal and estuarine flows** with advanced numerics, high flexibility, and easy extensibility.

Thetis builds on the [Firedrake](https://www.firedrakeproject.org/) finite element library to provide robust solvers for 2D/3D shallow water equations and related physical processes in the coastal ocean.

---

## 🌟 Key Features

✅ 2D depth-averaged and 3D baroclinic shallow water solvers  
✅ Wetting and drying schemes for realistic coastlines  
✅ Scalar transport (salinity, temperature, tracers)  
✅ Adjoint capabilities for data assimilation and inverse modeling  
✅ Flexible unstructured meshes  
✅ Open-source and easily extensible Python codebase

> 📚 Full documentation and installation instructions are available at [thetisproject.org](https://thetisproject.org/)

---

## 🚀 Installation


Please see the [installation instructions on the website](https://thetisproject.org/installation.html) for full details.

For the release branch:

1. Install **[Firedrake](https://www.firedrakeproject.org/install.html)** and activate its virtual environment.  
2. Install Thetis within the Firedrake environment:

```bash
pip install git+https://github.com/thetisproject/thetis.git
```
You can instead install and switch between different versions of Thetis with:

```bash
git clone https://github.com/thetisproject/thetis
cd thetis
git checkout <branch_name>
pip install -e .
```
> **Note**
> The latest development branch of Thetis (master) will require the corresponding main branch of Firedrake. See the
> website for more details.

### Release branch weekly status
[![Weekly Tests (release)](https://github.com/thetisproject/thetis/actions/workflows/weekly-release.yml/badge.svg?branch=release)](https://github.com/thetisproject/thetis/actions/workflows/weekly-release.yml)

---

## Getting Started

To get up and running with Thetis once it has been installed, we recommend checking out the [basic tutorials and 
documentation](https://thetisproject.org/documentation.html#tutorials). Further examples can then be found in the 
repository at [`examples`](./examples). See [`examples/README.md`](./examples/readme.md) for 
detailed descriptions of each example script.


---

## 📬 **Questions?**  

Check the [website](https://thetisproject.org/contact.html) for ways of reaching out to developers!

---

## Citing Thetis

If Thetis is helpful in your research, please cite:

> Kärnä, T., Kramer, S. C., Mitchell, L., Ham, D. A., Piggott, M. D. & Baptista, A. M. (2018).  
> *Thetis coastal ocean model: discontinuous Galerkin discretization for the three-dimensional hydrostatic equations.*  
> Geoscientific Model Development, 11: 4359–4382.  
> doi: [10.5194/gmd-11-4359-2018](https://doi.org/10.5194/gmd-11-4359-2018)

---


## License

Thetis is available under the MIT License. See the [LICENSE](./LICENSE) file for details.

