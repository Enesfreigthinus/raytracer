# RayTracer v4 - High-Performance C++ Ray Tracer

This is **Version 4** of my custom-built Ray Tracer written in modern **C++17**, optimized for rendering scenes with thousands of spheres using **multithreading** and **uniform spatial grid acceleration**.

---

## Features

- Ray-sphere intersection with diffuse shading
- Procedural scene generation (~15,000 non-overlapping spheres)
- Multithreaded rendering with `std::thread`
- Uniform Grid acceleration structure for fast intersection testing
- Mutex-safe grid writes to avoid race conditions
- Optimized with `-O3 -march=native` for performance
- Outputs PPM image (`output_grid.ppm`)
- Fully memory-safe (Valgrind-verified)

---

## Technologies Used

- **C++17** (Standard Library only — no third-party libraries)
- **Multithreading**
- **Uniform Grid** (spatial subdivision)
- **Valgrind** and **mpstat** for memory and CPU usage analysis

---

## Build and Run (Linux)

```bash
g++ -O3 -march=native -std=c++17 -pthread raytracer_v4.cpp -o raytracer-v4
./raytracer-v4 > output_grid.ppm
