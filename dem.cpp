// ---------------------------------------------------------------
//  dem.cpp  -  3-D Discrete Element Method solver (serial)
//  HPSC 2026, Assignment 1
//  Harsh Pandey | B23441 | SMME, IIT Mandi
//
//  Contact model : linear spring-dashpot
//  Integrator    : semi-implicit (symplectic) Euler
//  Contacts      : particle-particle + all 6 box walls
// ---------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <cassert>

// -- Vec3: minimal 3-D vector type with basic arithmetic --------
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3  operator+ (const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3  operator- (const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3  operator* (double s)      const { return {x*s,   y*s,   z*s};   }
    Vec3  operator/ (double s)      const { return {x/s,   y/s,   z/s};   }
    Vec3& operator+=(const Vec3& o)       { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o)       { x-=o.x; y-=o.y; z-=o.z; return *this; }

    double dot  (const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    double norm ()              const { return std::sqrt(dot(*this)); }
    // returns zero vector if nearly zero length to avoid divide-by-zero
    Vec3   unit ()              const { double n = norm(); return (n > 1e-14) ? (*this)/n : Vec3{}; }
};
inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// -- Particle: position, velocity, accumulated force, mass, radius
struct Particle {
    Vec3   pos;
    Vec3   vel;
    Vec3   force;        // reset each timestep
    double mass   = 1.0;
    double radius = 0.05;
};

// -- SimParams: everything that controls a run in one place -----
struct SimParams {
    // box dimensions
    double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    // contact model
    double kn    = 1e5;   // spring stiffness  [N/m]
    double gamma = 50.0;  // dashpot coefficient [N·s/m]

    Vec3   g     = {0.0, 0.0, -9.81};  // gravity [m/s^2]

    // time stepping
    double dt      = 1e-5;
    double t_total = 1.0;

    // output
    int    out_freq   = 1000;
    bool   write_traj = true;
};

// -- timing helper ----------------------------------------------
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
inline double elapsed_ms(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// -- zero_forces: clear accumulators before each timestep -------
void zero_forces(std::vector<Particle>& P) {
    for (auto& p : P) p.force = {0.0, 0.0, 0.0};
}

// -- add_gravity: F += m*g for every particle -------------------
void add_gravity(std::vector<Particle>& P, const Vec3& g) {
    for (auto& p : P) p.force += p.mass * g;
}

// -- spring-dashpot normal force, clamped to zero on separation -
inline double contact_force_magnitude(double overlap, double vn_rel,
                                      double kn, double gamma_n) {
    double Fn = kn * overlap - gamma_n * vn_rel;
    return (Fn > 0.0) ? Fn : 0.0;
}

// -- compute_particle_contacts: O(N^2) all-pairs loop -----------
void compute_particle_contacts(std::vector<Particle>& P,
                                double kn, double gamma_n) {
    int N = static_cast<int>(P.size());
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec3 rij   = P[j].pos - P[i].pos;
            double dij = rij.norm();

            double delta = P[i].radius + P[j].radius - dij;
            if (delta <= 0.0) continue;

            Vec3   nij  = rij.unit();
            double vn   = (P[j].vel - P[i].vel).dot(nij);

            double Fn   = contact_force_magnitude(delta, vn, kn, gamma_n);
            Vec3   Fc   = Fn * nij;

            P[i].force -= Fc;   // push i away from j
            P[j].force += Fc;   // push j away from i
        }
    }
}

// -- compute_wall_contacts: six faces of the bounding box -------
void compute_wall_contacts(std::vector<Particle>& P, const SimParams& sp) {
    for (auto& p : P) {
        double R = p.radius;

        // floor (normal = +z)
        {
            double delta = R - p.pos.z;
            if (delta > 0.0) {
                double vn = p.vel.z;
                double Fn = contact_force_magnitude(delta, vn, sp.kn, sp.gamma);
                p.force.z += Fn;
            }
        }
        // ceiling (normal = -z)
        {
            double delta = p.pos.z + R - sp.Lz;
            if (delta > 0.0) {
                double vn = -p.vel.z;
                double Fn = contact_force_magnitude(delta, vn, sp.kn, sp.gamma);
                p.force.z -= Fn;
            }
        }
        // x-min wall
        {
            double delta = R - p.pos.x;
            if (delta > 0.0) {
                double vn = p.vel.x;
                double Fn = contact_force_magnitude(delta, vn, sp.kn, sp.gamma);
                p.force.x += Fn;
            }
        }
        // x-max wall
        {
            double delta = p.pos.x + R - sp.Lx;
            if (delta > 0.0) {
                double vn = -p.vel.x;
                double Fn = contact_force_magnitude(delta, vn, sp.kn, sp.gamma);
                p.force.x -= Fn;
            }
        }
        // y-min wall
        {
            double delta = R - p.pos.y;
            if (delta > 0.0) {
                double vn = p.vel.y;
                double Fn = contact_force_magnitude(delta, vn, sp.kn, sp.gamma);
                p.force.y += Fn;
            }
        }
        // y-max wall
        {
            double delta = p.pos.y + R - sp.Ly;
            if (delta > 0.0) {
                double vn = -p.vel.y;
                double Fn = contact_force_magnitude(delta, vn, sp.kn, sp.gamma);
                p.force.y -= Fn;
            }
        }
    }
}

// -- semi-implicit Euler: v first, then x using new v -----------
void integrate_particles(std::vector<Particle>& P, double dt) {
    for (auto& p : P) {
        p.vel += (p.force / p.mass) * dt;   // v^{n+1}
        p.pos += p.vel * dt;                 // x^{n+1} uses updated v
    }
}

// -- kinetic energy sum -----------------------------------------
double compute_kinetic_energy(const std::vector<Particle>& P) {
    double KE = 0.0;
    for (const auto& p : P)
        KE += 0.5 * p.mass * p.vel.dot(p.vel);
    return KE;
}

// -- write one CSV line: time, KE, then positions+velocities ----
void write_output(std::ofstream& ofs, double t,
                  const std::vector<Particle>& P, double KE) {
    ofs << std::fixed << std::setprecision(8) << t << "," << KE;
    for (const auto& p : P)
        ofs << "," << p.pos.x << "," << p.pos.y << "," << p.pos.z
            << "," << p.vel.x << "," << p.vel.y << "," << p.vel.z;
    ofs << "\n";
}

// -- profiling counters per phase -------------------------------
struct ProfileData {
    double t_gravity   = 0.0;
    double t_p_contact = 0.0;
    double t_w_contact = 0.0;
    double t_integrate = 0.0;
    double t_other     = 0.0;
    double t_total     = 0.0;
};

// -- main time loop; returns profiling data ---------------------
ProfileData run_simulation(std::vector<Particle>& P,
                            const SimParams& sp,
                            const std::string& outfile = "") {
    ProfileData prof;

    std::ofstream ofs;
    if (!outfile.empty() && sp.write_traj) {
        ofs.open(outfile);
        ofs << "t,KE";
        for (int i = 0; i < (int)P.size(); ++i)
            ofs << ",x" << i << ",y" << i << ",z" << i
                << ",vx" << i << ",vy" << i << ",vz" << i;
        ofs << "\n";
    }

    int    nsteps = static_cast<int>(sp.t_total / sp.dt);
    double t      = 0.0;

    auto t_sim_start = Clock::now();

    for (int step = 0; step < nsteps; ++step) {
        zero_forces(P);

        auto t0 = Clock::now();
        add_gravity(P, sp.g);
        prof.t_gravity += elapsed_ms(t0);

        t0 = Clock::now();
        compute_particle_contacts(P, sp.kn, sp.gamma);
        prof.t_p_contact += elapsed_ms(t0);

        t0 = Clock::now();
        compute_wall_contacts(P, sp);
        prof.t_w_contact += elapsed_ms(t0);

        t0 = Clock::now();
        integrate_particles(P, sp.dt);
        prof.t_integrate += elapsed_ms(t0);

        t += sp.dt;

        if (ofs.is_open() && step % sp.out_freq == 0) {
            auto t1 = Clock::now();
            double KE = compute_kinetic_energy(P);
            write_output(ofs, t, P, KE);
            prof.t_other += elapsed_ms(t1);
        }
    }

    prof.t_total = elapsed_ms(t_sim_start);
    return prof;
}

// ==============================================================
//  Test 1: free fall - single particle, no damping
//  z(t) = z0 - 0.5*g*t^2
// ==============================================================
void test_free_fall() {
    std::cout << "\n=== TEST 1: Free Fall ===\n";

    SimParams sp;
    sp.Lx = 20.0; sp.Ly = 20.0; sp.Lz = 20.0;
    sp.kn      = 1e5;
    sp.gamma   = 0.0;
    sp.dt      = 1e-4;
    sp.t_total = 1.2;
    sp.out_freq = 1;
    sp.write_traj = true;

    double z0 = 8.0, R = 0.5;
    Particle p0;
    p0.pos    = {10.0, 10.0, z0};
    p0.vel    = {0.0,  0.0,  0.0};
    p0.mass   = 1.0;
    p0.radius = R;

    std::vector<Particle> particles = {p0};
    run_simulation(particles, sp, "test1_freefall.csv");

    double t_check = 1.0;
    double z_anal  = z0 - 0.5 * 9.81 * t_check * t_check;
    std::cout << "  At t = " << t_check << " s:\n"
              << "  Analytical z = " << z_anal << " m\n"
              << "  (See test1_freefall.csv for full trajectory)\n";
}

// ==============================================================
//  Test 2: constant velocity - no gravity, straight-line motion
// ==============================================================
void test_constant_velocity() {
    std::cout << "\n=== TEST 2: Constant Velocity ===\n";

    SimParams sp;
    sp.Lx = 20.0; sp.Ly = 20.0; sp.Lz = 20.0;
    sp.g       = {0.0, 0.0, 0.0};
    sp.gamma   = 0.0;
    sp.dt      = 1e-4;
    sp.t_total = 0.5;
    sp.out_freq = 1;
    sp.write_traj = true;

    double vx0 = 4.0;
    Particle p0;
    p0.pos    = {5.0, 10.0, 5.0};
    p0.vel    = {vx0, 0.0, 0.0};
    p0.mass   = 1.0;
    p0.radius = 0.05;

    double x0 = p0.pos.x;
    std::vector<Particle> particles = {p0};
    run_simulation(particles, sp, "test2_constvel.csv");

    double t_final = sp.t_total;
    double x_anal  = x0 + vx0 * t_final;
    double x_num   = particles[0].pos.x;
    std::cout << "  x analytical = " << x_anal << " m\n"
              << "  x numerical  = " << x_num  << " m\n"
              << "  error        = " << std::abs(x_num - x_anal) << " m\n";
    assert(std::abs(x_num - x_anal) < 1e-10 && "Constant velocity test FAILED");
    std::cout << "  PASSED (machine-precision agreement)\n";
}

// ==============================================================
//  Test 3: bouncing particle with damping
// ==============================================================
void test_bounce() {
    std::cout << "\n=== TEST 3: Bouncing Particle ===\n";

    SimParams sp;
    sp.Lx = 20.0; sp.Ly = 20.0; sp.Lz = 20.0;
    sp.kn      = 1e5;
    sp.gamma   = 50.0;
    sp.dt      = 1e-5;
    sp.t_total = 3.0;
    sp.out_freq = 100;
    sp.write_traj = true;

    Particle p0;
    p0.pos    = {10.0, 10.0, 5.0};
    p0.vel    = {0.0, 0.0, 0.0};
    p0.mass   = 1.0;
    p0.radius = 0.5;

    std::vector<Particle> particles = {p0};
    run_simulation(particles, sp, "test3_bounce.csv");

    std::cout << "  Final position z = " << particles[0].pos.z << " m\n"
              << "  Final KE        = " << compute_kinetic_energy(particles) << " J\n"
              << "  (See test3_bounce.csv for height and KE vs time)\n";
}

// ==============================================================
//  Profiling run: time each phase for N particles
// ==============================================================
void run_profiling(int N) {
    std::cout << "\n=== PROFILING: N = " << N << " particles ===\n";

    SimParams sp;
    sp.Lx = 2.0; sp.Ly = 2.0; sp.Lz = 2.0;
    sp.kn       = 1e5;
    sp.gamma    = 50.0;
    sp.dt       = 1e-5;
    sp.t_total  = 0.01;
    sp.out_freq = 1000000;  // skip output during profiling
    sp.write_traj = false;

    std::vector<Particle> particles;
    particles.reserve(N);
    double R = 0.05;
    int    nx = static_cast<int>(std::cbrt(N)) + 1;
    double spacing = (sp.Lx - 2.0 * R) / nx;
    int    count = 0;
    for (int ix = 0; ix < nx && count < N; ++ix)
    for (int iy = 0; iy < nx && count < N; ++iy)
    for (int iz = 0; iz < nx && count < N; ++iz) {
        Particle p;
        p.pos    = {R + ix * spacing + 1e-4,
                    R + iy * spacing + 1e-4,
                    R + iz * spacing + 1e-4};
        p.vel    = {0.0, 0.0, 0.0};
        p.mass   = 1.0;
        p.radius = R;
        particles.push_back(p);
        ++count;
    }

    ProfileData prof = run_simulation(particles, sp, "");

    auto pct = [&](double t) { return 100.0 * t / prof.t_total; };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total wall time      : " << prof.t_total    << " ms\n";
    std::cout << "  Particle contacts    : " << pct(prof.t_p_contact) << " %\n";
    std::cout << "  Wall contacts        : " << pct(prof.t_w_contact) << " %\n";
    std::cout << "  Integration          : " << pct(prof.t_integrate) << " %\n";
    std::cout << "  Gravity              : " << pct(prof.t_gravity)   << " %\n";
    std::cout << "  Other (I/O, KE)      : " << pct(prof.t_other)     << " %\n";

    std::string fname = "profile_N" + std::to_string(N) + ".csv";
    std::ofstream pf(fname);
    pf << "phase,fraction\n";
    pf << "Particle contacts,"  << pct(prof.t_p_contact) << "\n";
    pf << "Wall contacts,"      << pct(prof.t_w_contact) << "\n";
    pf << "Integration,"        << pct(prof.t_integrate) << "\n";
    pf << "Gravity,"            << pct(prof.t_gravity)   << "\n";
    pf << "Other,"              << pct(prof.t_other)     << "\n";
    pf.close();
    std::cout << "  Profiling CSV -> " << fname << "\n";
}

// ==============================================================
//  Multi-particle settling run
// ==============================================================
void run_multiparticle(int N, double t_total, const std::string& tag) {
    std::cout << "\n=== MULTI-PARTICLE: N = " << N << " ===\n";

    SimParams sp;
    sp.Lx = 1.0; sp.Ly = 1.0; sp.Lz = 2.0;
    sp.kn      = 1e5;
    sp.gamma   = 50.0;
    sp.dt      = 1e-5;
    sp.t_total = t_total;
    sp.out_freq = static_cast<int>(0.01 / sp.dt);  // every 10 ms
    sp.write_traj = true;

    std::vector<Particle> particles;
    particles.reserve(N);
    double R = 0.05;
    int    nx = static_cast<int>(std::cbrt(N)) + 1;
    double spacing = 0.15;

    int count = 0;
    for (int ix = 0; ix < nx && count < N; ++ix)
    for (int iy = 0; iy < nx && count < N; ++iy)
    for (int iz = 0; iz < nx && count < N; ++iz) {
        Particle p;
        p.pos    = {R + 0.01 + ix * spacing,
                    R + 0.01 + iy * spacing,
                    0.5 + iz * spacing};
        p.vel    = {0.0, 0.0, 0.0};
        p.mass   = 1.0;
        p.radius = R;
        particles.push_back(p);
        ++count;
    }

    auto tstart = Clock::now();
    run_simulation(particles, sp, "sim_N" + tag + ".csv");
    double elapsed = elapsed_ms(tstart);

    std::cout << "  Wall time: " << elapsed << " ms\n";
    std::cout << "  Final KE : " << compute_kinetic_energy(particles) << " J\n";
}

// ==============================================================
//  main
// ==============================================================
int main() {
    std::cout << "==============================================\n";
    std::cout << "  DEM Solver  -  Serial Version\n";
    std::cout << "==============================================\n";

    test_free_fall();
    test_constant_velocity();
    test_bounce();

    run_profiling(200);
    run_profiling(1000);
    run_profiling(5000);

    run_multiparticle(200,  0.5, "200");
    run_multiparticle(1000, 0.5, "1000");
    // run_multiparticle(5000, 0.5, "5000");  // uncomment when ready

    std::cout << "\nAll done. CSV output files written.\n";
    return 0;
}
