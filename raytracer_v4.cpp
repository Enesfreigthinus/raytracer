// -----------------------------------------------------------------------------
// raytracer_v3_grid.cpp  –  Çakışmasız küre yerleşimi + multithread render
// g++ -O3 -march=native -std=c++17 -pthread raytracer_v3_grid.cpp -o rt3
// -----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif


constexpr int W = 720, H = 480, SPP = 16, N_SPHERES = 15000;

//-----------------------------------------------------
// 1. Mini Vec3
//-----------------------------------------------------
struct Vec3 {
    double x, y, z;
    Vec3(double a = 0, double b = 0, double c = 0) : x(a), y(b), z(c) {}
    Vec3 operator-() const { return { -x, -y, -z }; }
    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator*=(double t) { x *= t; y *= t; z *= t; return *this; }
    Vec3& operator/=(double t) { return *this *= 1 / t; }
    double length()  const { return std::sqrt(x * x + y * y + z * z); }
    double length2() const { return x * x + y * y + z * z; }
};
using Point3 = Vec3; using Color = Vec3;

inline Vec3 operator+(Vec3 a, const Vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator*(Vec3 a, const Vec3& b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
inline Vec3 operator*(double t, Vec3 v)      { return { t * v.x, t * v.y, t * v.z }; }
inline Vec3 operator*(Vec3 v, double t)      { return t * v; }
inline Vec3 operator/(Vec3 v, double t)      { return (1 / t) * v; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline Vec3 unit(Vec3 v) { return v / v.length(); }


// 2. RNG – hızlı XorShift64 (thread-safe)

struct XorShift64 {
    uint64_t s;
    explicit XorShift64(uint64_t seed) { s = seed ? seed : 0xdeadbeefULL; }
    inline double next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (s >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
    }
};
thread_local XorShift64 rng{ 0x9e3779b97f4a7c15ULL ^
                             std::hash<std::thread::id>{}(std::this_thread::get_id()) };
inline double rnd() { return rng.next(); }

//-----------------------------------------------------
// 3. Ray & Camera
//-----------------------------------------------------
struct Ray { Point3 orig; Vec3 dir; Point3 at(double t) const { return orig + t * dir; } };

struct Camera {
    Point3 orig; Vec3 horiz, vert, llc;
    Camera(Point3 lookfrom, Point3 lookat, Vec3 vup,
           double vfov, double aspect) {
        double theta = vfov * M_PI / 180.0;
        double h = std::tan(theta / 2);
        double vh = 2 * h, vw = aspect * vh;
        Vec3 w = unit(lookfrom - lookat);
        Vec3 u = unit(cross(vup, w));
        Vec3 v = cross(w, u);
        orig = lookfrom;
        horiz = vw * u;   vert = vh * v;
        llc = orig - horiz / 2 - vert / 2 - w;
    }
    Ray get_ray(double s, double t) const { return { orig, llc + s * horiz + t * vert - orig }; }
};

//-----------------------------------------------------
// 4. Dünya ve küreler
//-----------------------------------------------------
struct Sphere { Point3 c; double r, r2; Color alb; };

struct World {
    std::vector<Sphere> obj;                 // Küre listesi
    bool hit(const Ray& ray, double tmin, double tmax,
             Point3& p, Vec3& n, Color& alb) const {
        bool any = false; double closest = tmax;
        double a = ray.dir.length2();
        for (const auto& s : obj) {
            Vec3 oc = ray.orig - s.c;
            double half_b = dot(oc, ray.dir);
            double c_ = oc.length2() - s.r2;
            double disc = half_b * half_b - a * c_;
            if (disc < 0) continue;
            double sd = std::sqrt(disc);
            double root = (-half_b - sd) / a;
            if (root < tmin || root > closest) {
                root = (-half_b + sd) / a;
                if (root < tmin || root > closest) continue;
            }
            any = true; closest = root;
            p = ray.at(root); n = (p - s.c) / s.r; alb = s.alb;
        }
        return any;
    }
};

//-----------------------------------------------------
// 5. Küre renkleri (HSV → RGB)
//-----------------------------------------------------
Color hsv_to_rgb(double h, double s, double v) {
    double c = v * s;
    double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
    double m = v - c;
    double r, g, b;
    if (h < 60)      { r = c; g = x; b = 0; }
    else if (h < 120){ r = x; g = c; b = 0; }
    else if (h < 180){ r = 0; g = c; b = x; }
    else if (h < 240){ r = 0; g = x; b = c; }
    else if (h < 300){ r = x; g = 0; b = c; }
    else             { r = c; g = 0; b = x; }
    return { r + m, g + m, b + m };
}

//-----------------------------------------------------
// 6. Shading
//-----------------------------------------------------
Color ray_color(const Ray& r, const World& w) {
    Point3 p; Vec3 n; Color alb;
    if (w.hit(r, 0.001, std::numeric_limits<double>::infinity(), p, n, alb)) {
        Vec3 l1 = unit(Vec3{ 0.6, 0.8,  0.3 });
        Vec3 l2 = unit(Vec3{-0.3, 0.5, -0.8 });
        double diff1 = std::max(0.0, dot(n, l1));
        double diff2 = std::max(0.0, dot(n, l2));
        double ambient = 0.20, main_l = 0.70 * diff1, fill_l = 0.40 * diff2;
        double light = std::min(1.0, ambient + main_l + fill_l);
        return light * alb;
    }
    Vec3 u = unit(r.dir); double t = 0.5 * (u.y + 1);
    Color sky_h{0.92, 0.97, 1.05}, sky_t{0.68, 0.78, 1.05};
    return (1 - t) * sky_h + t * sky_t;
}
inline int to8(double x) {
    double g = std::pow(std::clamp(x, 0.0, 1.0), 1.0 / 2.2);
    return int(255.0 * g);
}

//-----------------------------------------------------
// 7. Uniform grid ayarları
//-----------------------------------------------------
constexpr double R_MAX = 0.45;
constexpr double GAP   = 0.05;
constexpr double CELL  = R_MAX*2 + GAP;            // 1.0
const Vec3  P_MIN{-12, 0, -12};
const Vec3  P_MAX{ 12, 6,  12};

// Hesaplanacak değerler
const Vec3 SCENE_SIZE{ P_MAX.x - P_MIN.x,
                       P_MAX.y - P_MIN.y,
                       P_MAX.z - P_MIN.z };

const int NX = int(std::ceil(SCENE_SIZE.x / CELL));
const int NY = int(std::ceil(SCENE_SIZE.y / CELL));
const int NZ = int(std::ceil(SCENE_SIZE.z / CELL));
const int N_CELL = NX * NY * NZ;

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

auto cell_index = [](const Point3& p) {
    int ix = clampi(int((p.x - P_MIN.x) / CELL), 0, NX - 1);
    int iy = clampi(int((p.y - P_MIN.y) / CELL), 0, NY - 1);
    int iz = clampi(int((p.z - P_MIN.z) / CELL), 0, NZ - 1);
    return (iz * NY + iy) * NX + ix;
};

//-----------------------------------------------------
// 8. Küre yerleştirme – paralel grid alg.
//-----------------------------------------------------
struct Grid {
    std::vector<std::vector<int>> cell;
    Grid(int n) : cell(n) {}
};

World world;
Grid  grid(N_CELL);
std::vector<std::mutex> cell_mx(N_CELL);
std::mutex world_mx;

bool try_insert(Sphere&& sp) {
    int cid = cell_index(sp.c);

    // Komşu hücrelerde çakışma var mı?
    int cx = cid % NX;
    int cy = (cid / NX) % NY;
    int cz = cid / (NX * NY);

    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= NX || ny < 0 || ny >= NY || nz < 0 || nz >= NZ) continue;
                int nid = (nz * NY + ny) * NX + nx;
                std::lock_guard<std::mutex> lk(cell_mx[nid]);
                for (int idx : grid.cell[nid]) {
                    const auto& s = world.obj[idx];
                    double lim = sp.r + s.r + GAP;
                    if ((sp.c - s.c).length2() < lim*lim) return false;
                }
            }

    {
        std::scoped_lock lk(world_mx, cell_mx[cid]);   // İKİ kilit aynı anda

       if (world.obj.size() >= N_SPHERES) return false;   // hedefe ulaştık

        int my_idx = world.obj.size();
        world.obj.push_back(std::move(sp));            // 1) küreyi ekle
        grid.cell[cid].push_back(my_idx);              // 2) index’i yaz
    }
    return true;
}

//-----------------------------------------------------
// 9. CPU affinity

#ifdef __linux__
void set_cpu_affinity(int tid) {
    cpu_set_t cpuset; CPU_ZERO(&cpuset);
    int nc = sysconf(_SC_NPROCESSORS_ONLN); //kaç logic core var
    CPU_SET(tid % nc, &cpuset);   //thread id'ye denk gelen çekirdeği seç
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
#endif

//-----------------------------------------------------
// 10. Main
//-----------------------------------------------------
int main() {
    
    const double aspect = double(W) / H;
    Camera cam({ 0,5,28 }, { 0,2,0 }, { 0,1,0 }, 35, aspect);

    std::cout << "Target spheres  : " << N_SPHERES << "\n";
    std::cout << "Grid dimensions : " << NX << "×" << NY << "×" << NZ
              << "  (" << N_CELL << " cells)\n";

    //------------------- Küre üretimi -------------------
    auto gen_worker = [&](int tid) {
#ifdef __linux__
        set_cpu_affinity(tid);
#endif
        std::mt19937_64 g(1337 + tid * 911);
        std::uniform_real_distribution<double> px(P_MIN.x, P_MAX.x);
        std::uniform_real_distribution<double> py(P_MIN.y, P_MAX.y);
        std::uniform_real_distribution<double> pz(P_MIN.z, P_MAX.z);
        std::uniform_real_distribution<double> pr(0.20, 0.45);
        std::uniform_real_distribution<double> h(0, 360), s(0.4, 0.9), v(0.5, 0.9);

        while (true) {
            if (world.obj.size() >= N_SPHERES) break;

            Sphere sp;
            sp.c = { px(g), py(g), pz(g) };
            sp.r = pr(g); sp.r2 = sp.r * sp.r;
            sp.alb = hsv_to_rgb(h(g), s(g), v(g));

            try_insert(std::move(sp));
        }
    };

    unsigned NT_gen = std::max(2u, std::thread::hardware_concurrency());
    std::cout << "Generating spheres with " << NT_gen << " threads…\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::thread> tg;
        for (unsigned t = 0; t < NT_gen; ++t) tg.emplace_back(gen_worker, t);
        for (auto& th : tg) th.join();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Placed " << world.obj.size() << " spheres in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    //------------------- Framebuffer -------------------
    std::vector<Color> fb(W * H);

    //------------------- Render (row-steal) -------------
    unsigned NT = std::max(2u, std::thread::hardware_concurrency());
    std::atomic<int> next_row{ 0 };
    std::cout << "Rendering with " << NT << " threads…\n";
    auto t2 = std::chrono::high_resolution_clock::now();

    auto worker = [&](int tid) {
#ifdef __linux__
        set_cpu_affinity(tid);
#endif
        int j;
        while ((j = next_row.fetch_add(1)) < H) {
            for (int i = 0; i < W; ++i) {
                Color c{ 0,0,0 };
                for (int s = 0; s < SPP; ++s) {
                    double u = (i + rnd()) / (W - 1);
                    double v = (j + rnd()) / (H - 1);
                    c += ray_color(cam.get_ray(u, v), world);
                }
                fb[j * W + i] = c / double(SPP);
            }
        }
    };

    {
        std::vector<std::thread> tr;
        for (unsigned t = 0; t < NT; ++t) tr.emplace_back(worker, t);
        for (auto& th : tr) th.join();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Render time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms\n";

    //------------------- PPM çıkışı ---------------------
    std::ofstream out("output_grid.ppm");
    out << "P3\n" << W << ' ' << H << "\n255\n";
    for (int j = H - 1; j >= 0; --j)
        for (int i = 0; i < W; ++i) {
            const auto& c = fb[j * W + i];
            out << to8(c.x) << ' ' << to8(c.y) << ' ' << to8(c.z) << '\n';
        }
    std::cout << "✓ output_grid.ppm yazıldı\n";
}
