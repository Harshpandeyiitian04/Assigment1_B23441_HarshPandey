// ---------------------------------------------------------------
//  dem_omp.cpp  -  3-D DEM solver, OpenMP parallel version
//  HPSC 2026, Assignment 1
//  Harsh Pandey | B23441 | SMME, IIT Mandi
//
//  Contact model : linear spring-dashpot
//  Integrator    : semi-implicit (symplectic) Euler
//  Parallelism   : OpenMP with per-thread force buffers
//
//  Build: g++ -std=c++17 -O2 -fopenmp -o dem_omp dem_omp.cpp
//  Run:   OMP_NUM_THREADS=8 ./dem_omp
// ---------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <omp.h>

// -- Vec3: lightweight 3-D vector -------------------------------
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
    Vec3   unit ()              const { double n=norm(); return (n>1e-14) ? (*this)/n : Vec3{}; }
};
inline Vec3 operator*(double s, const Vec3& v) { return v*s; }

// -- Particle state ---------------------------------------------
struct Particle {
    Vec3   pos;
    Vec3   vel;
    Vec3   force;
    double mass   = 1.0;
    double radius = 0.05;
};

// -- simulation parameters -------------------------------------
struct SimParams {
    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    double kn    = 1e5;
    double gamma = 50.0;
    Vec3   g     = {0.0, 0.0, -9.81};
    double dt      = 1e-5;
    double t_total = 1.0;
    int    out_freq   = 1000;
    bool   write_traj = true;
};

// -- timing helpers --------------------------------------------
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
inline double elapsed_ms(TimePoint t0){
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

// -- spring-dashpot force, clamped non-negative ----------------
inline double contact_Fn(double overlap, double vn, double kn, double gn){
    double Fn = kn*overlap - gn*vn;
    return (Fn > 0.0) ? Fn : 0.0;
}

// -- zero forces: each particle is independent so this is easy to parallelise
void zero_forces(std::vector<Particle>& P){
    int N = (int)P.size();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i) P[i].force = {0.0,0.0,0.0};
}

// -- gravity: also trivially parallel --------------------------
void add_gravity(std::vector<Particle>& P, const Vec3& g){
    int N = (int)P.size();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i) P[i].force += P[i].mass * g;
}

// -- compute_particle_contacts: parallelised with per-thread force buffers
//
//  The naive approach of putting a parallel for on the outer i-loop
//  creates a race condition: two threads can both try to write to
//  P[j].force for the same j at the same time.
//
//  Fix: give every thread its own local force array (lf[tid]).
//  Each thread only writes to lf[tid], so no synchronisation is needed
//  inside the loop. After the parallel region a serial reduction sums
//  the buffers into P[i].force.  Cost of reduction is O(N*p), which
//  is negligible compared to the O(N^2) contact work.
//
//  Dynamic scheduling with chunk=32 because the outer i-loop has
//  triangular work (i=0 does N-1 inner steps, i=N-2 does just 1).
//  Static would leave the high-i threads sitting idle.
// --------------------------------------------------------------
void compute_particle_contacts(std::vector<Particle>& P,
                                double kn, double gn,
                                std::vector<std::vector<Vec3>>& lf,
                                int nthreads)
{
    int N = (int)P.size();

    // zero the per-thread buffers
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for(int i=0;i<N;++i) lf[tid][i] = {0.0,0.0,0.0};
    }

    // parallel contact loop
    #pragma omp parallel
    {
        int tid   = omp_get_thread_num();
        auto& myf = lf[tid];

        #pragma omp for schedule(dynamic,32) nowait
        for(int i=0;i<N-1;++i){
            const Vec3& xi = P[i].pos;
            const Vec3& vi = P[i].vel;
            double Ri = P[i].radius;

            for(int j=i+1;j<N;++j){
                Vec3   rij  = P[j].pos - xi;
                double dij  = rij.norm();
                double delta = Ri + P[j].radius - dij;
                if(delta <= 0.0) continue;

                Vec3   nij  = rij.unit();
                double vn   = (P[j].vel - vi).dot(nij);
                double Fn   = contact_Fn(delta, vn, kn, gn);
                Vec3   Fc   = Fn * nij;

                myf[i] -= Fc;
                myf[j] += Fc;
            }
        }
    }  // implicit barrier

    // serial reduction: accumulate per-thread buffers into particle forces
    for(int t=0;t<nthreads;++t)
        for(int i=0;i<N;++i)
            P[i].force += lf[t][i];
}

// -- wall contacts: trivially parallel since each particle is independent
//
//  Sign convention: nhat_outward points INTO the domain.
//  Fn = max(0, kn*delta - gamma*vn), force = Fn * nhat_outward
// --------------------------------------------------------------
void compute_wall_contacts(std::vector<Particle>& P, const SimParams& sp){
    int N = (int)P.size();
    #pragma omp parallel for schedule(static)
    for(int idx=0;idx<N;++idx){
        auto& p  = P[idx];
        double R = p.radius;

        // floor (nhat = +z)
        { double delta = R - p.pos.z;
          if(delta > 0.0){ double vn = p.vel.z;
              p.force.z += contact_Fn(delta,vn,sp.kn,sp.gamma); }}

        // ceiling (nhat = -z)
        { double delta = p.pos.z + R - sp.Lz;
          if(delta > 0.0){ double vn = -p.vel.z;
              p.force.z -= contact_Fn(delta,vn,sp.kn,sp.gamma); }}

        // x-min wall (nhat = +x)
        { double delta = R - p.pos.x;
          if(delta > 0.0){ double vn = p.vel.x;
              p.force.x += contact_Fn(delta,vn,sp.kn,sp.gamma); }}

        // x-max wall (nhat = -x)
        { double delta = p.pos.x + R - sp.Lx;
          if(delta > 0.0){ double vn = -p.vel.x;
              p.force.x -= contact_Fn(delta,vn,sp.kn,sp.gamma); }}

        // y-min wall (nhat = +y)
        { double delta = R - p.pos.y;
          if(delta > 0.0){ double vn = p.vel.y;
              p.force.y += contact_Fn(delta,vn,sp.kn,sp.gamma); }}

        // y-max wall (nhat = -y)
        { double delta = p.pos.y + R - sp.Ly;
          if(delta > 0.0){ double vn = -p.vel.y;
              p.force.y -= contact_Fn(delta,vn,sp.kn,sp.gamma); }}
    }
}

// -- integration: trivially parallel ---------------------------
void integrate_particles(std::vector<Particle>& P, double dt){
    int N = (int)P.size();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){
        P[i].vel += (P[i].force / P[i].mass) * dt;
        P[i].pos += P[i].vel * dt;
    }
}

// -- parallel KE reduction -------------------------------------
double compute_kinetic_energy(const std::vector<Particle>& P){
    double KE = 0.0;
    int N = (int)P.size();
    #pragma omp parallel for schedule(static) reduction(+:KE)
    for(int i=0;i<N;++i)
        KE += 0.5 * P[i].mass * P[i].vel.dot(P[i].vel);
    return KE;
}

// -- CSV output ------------------------------------------------
void write_output(std::ofstream& ofs, double t,
                  const std::vector<Particle>& P, double KE){
    ofs << std::fixed << std::setprecision(8) << t << "," << KE;
    for(const auto& p : P)
        ofs << "," << p.pos.x << "," << p.pos.y << "," << p.pos.z
            << "," << p.vel.x << "," << p.vel.y << "," << p.vel.z;
    ofs << "\n";
}

// -- per-phase timing counters ---------------------------------
struct ProfileData {
    double t_gravity   = 0.0;
    double t_p_contact = 0.0;
    double t_w_contact = 0.0;
    double t_integrate = 0.0;
    double t_other     = 0.0;
    double t_total     = 0.0;
};

// -- main time loop --------------------------------------------
ProfileData run_simulation(std::vector<Particle>& P,
                            const SimParams& sp,
                            const std::string& outfile = "")
{
    ProfileData prof;
    int N        = (int)P.size();
    int nthreads = omp_get_max_threads();

    // per-thread force buffers: lf[thread_id][particle_id]
    std::vector<std::vector<Vec3>> lf(nthreads, std::vector<Vec3>(N));

    std::ofstream ofs;
    if(!outfile.empty() && sp.write_traj){
        ofs.open(outfile);
        ofs << "t,KE";
        for(int i=0;i<N;++i)
            ofs << ",x"<<i<<",y"<<i<<",z"<<i
                << ",vx"<<i<<",vy"<<i<<",vz"<<i;
        ofs << "\n";
    }

    int    nsteps = (int)(sp.t_total / sp.dt);
    double t      = 0.0;
    auto   t_sim  = Clock::now();

    for(int step=0; step<nsteps; ++step){

        zero_forces(P);

        auto t0 = Clock::now();
        add_gravity(P, sp.g);
        prof.t_gravity += elapsed_ms(t0);

        t0 = Clock::now();
        compute_particle_contacts(P, sp.kn, sp.gamma, lf, nthreads);
        prof.t_p_contact += elapsed_ms(t0);

        t0 = Clock::now();
        compute_wall_contacts(P, sp);
        prof.t_w_contact += elapsed_ms(t0);

        t0 = Clock::now();
        integrate_particles(P, sp.dt);
        prof.t_integrate += elapsed_ms(t0);

        t += sp.dt;

        if(ofs.is_open() && step % sp.out_freq == 0){
            auto t1 = Clock::now();
            double KE = compute_kinetic_energy(P);
            write_output(ofs, t, P, KE);
            prof.t_other += elapsed_ms(t1);
        }
    }

    prof.t_total = elapsed_ms(t_sim);
    return prof;
}

// -- place N particles on a regular grid inside the box --------
std::vector<Particle> make_grid(int N, const SimParams& sp){
    std::vector<Particle> P;
    P.reserve(N);
    double R       = 0.05;
    int    nx      = (int)std::cbrt((double)N) + 1;
    double spacing = (sp.Lx - 2.0*R) / nx;
    int    count   = 0;
    for(int ix=0; ix<nx && count<N; ++ix)
    for(int iy=0; iy<nx && count<N; ++iy)
    for(int iz=0; iz<nx && count<N; ++iz){
        Particle p;
        p.pos    = {R + ix*spacing + 1e-4,
                    R + iy*spacing + 1e-4,
                    R + iz*spacing + 1e-4};
        p.vel    = {0,0,0};
        p.mass   = 1.0;
        p.radius = R;
        P.push_back(p);
        ++count;
    }
    return P;
}

// ==============================================================
//  Correctness check: serial (p=1) vs parallel must agree to
//  machine precision on the same initial conditions.
// ==============================================================
void correctness_check(){
    std::cout << "\n=== CORRECTNESS CHECK (serial vs parallel) ===\n";

    SimParams sp;
    sp.Lx=1.0; sp.Ly=1.0; sp.Lz=1.0;
    sp.kn=1e5; sp.gamma=50.0; sp.dt=1e-5; sp.t_total=0.01;
    sp.out_freq = 100; sp.write_traj = true;

    auto P_serial = make_grid(50, sp);
    auto P_par    = P_serial;   // exact copy, same ICs

    omp_set_num_threads(1);
    run_simulation(P_serial, sp, "check_serial.csv");

    omp_set_num_threads(omp_get_max_threads());
    run_simulation(P_par, sp, "check_par.csv");

    double KE_s = compute_kinetic_energy(P_serial);
    double KE_p = compute_kinetic_energy(P_par);
    double rel_err = std::abs(KE_s - KE_p) / (KE_s + 1e-30);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  KE serial   = " << KE_s << " J\n"
              << "  KE parallel = " << KE_p << " J\n"
              << "  Relative error = " << rel_err << "\n";
    if(rel_err < 1e-10)
        std::cout << "  PASSED - results match to machine precision\n";
    else
        std::cout << "  WARNING - results differ! Check for race conditions.\n";
}

// ==============================================================
//  Scaling study: sweep N x p, write to scaling_results.csv
// ==============================================================
void scaling_study(){
    std::cout << "\n=== SCALING STUDY ===\n";

    std::vector<int> Ns       = {200, 1000, 5000};
    std::vector<int> threads  = {1, 2, 4, 8, 16};

    SimParams sp;
    sp.Lx=2.0; sp.Ly=2.0; sp.Lz=2.0;
    sp.kn=1e5; sp.gamma=50.0; sp.dt=1e-5;
    sp.t_total    = 0.01;
    sp.out_freq   = 1000000;
    sp.write_traj = false;

    std::ofstream csv("scaling_results.csv");
    csv << "N,threads,T1_ms,Tp_ms,speedup,efficiency\n";
    csv << std::fixed << std::setprecision(4);

    std::cout << std::setw(8)  << "N"
              << std::setw(10) << "threads"
              << std::setw(14) << "T1 (ms)"
              << std::setw(14) << "Tp (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Efficiency"
              << "\n";
    std::cout << std::string(70,'-') << "\n";

    for(int N : Ns){
        omp_set_num_threads(1);
        auto P0  = make_grid(N, sp);
        auto pr1 = run_simulation(P0, sp);
        double T1 = pr1.t_total;

        for(int p : threads){
            omp_set_num_threads(p);
            auto P   = make_grid(N, sp);
            auto pr  = run_simulation(P, sp);
            double Tp = pr.t_total;
            double S  = T1 / Tp;
            double E  = S  / p;

            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(8)  << N
                      << std::setw(10) << p
                      << std::setw(14) << T1
                      << std::setw(14) << Tp
                      << std::setw(12) << S
                      << std::setw(12) << E
                      << "\n";

            csv << N << "," << p << "," << T1 << "," << Tp
                << "," << S << "," << E << "\n";
        }
        std::cout << "\n";
    }

    csv.close();
    std::cout << "  Results written to scaling_results.csv\n";
}

// ==============================================================
//  Verification tests (run serial to compare against analytical)
// ==============================================================
void test_freefall(){
    std::cout << "\n=== TEST 1: Free Fall ===\n";
    SimParams sp;
    sp.Lx=20; sp.Ly=20; sp.Lz=20;
    sp.kn=1e5; sp.gamma=0.0; sp.dt=1e-4; sp.t_total=1.2;
    sp.out_freq=1; sp.write_traj=true;

    Particle p0;
    p0.pos={10,10,8}; p0.vel={0,0,0}; p0.mass=1; p0.radius=0.5;
    std::vector<Particle> P={p0};
    omp_set_num_threads(1);
    run_simulation(P, sp, "omp_test1_freefall.csv");

    double t=1.0, z_anal=8.0-0.5*9.81*t*t;
    std::cout << "  z_analytical(t=1s) = " << z_anal << " m\n";
    std::cout << "  (compare with omp_test1_freefall.csv)\n";
}

void test_bounce(){
    std::cout << "\n=== TEST 3: Bouncing Particle ===\n";
    SimParams sp;
    sp.Lx=20; sp.Ly=20; sp.Lz=20;
    sp.kn=1e5; sp.gamma=50.0; sp.dt=1e-5; sp.t_total=3.0;
    sp.out_freq=100; sp.write_traj=true;

    Particle p0;
    p0.pos={10,10,5}; p0.vel={0,0,0}; p0.mass=1; p0.radius=0.5;
    std::vector<Particle> P={p0};
    omp_set_num_threads(1);
    run_simulation(P, sp, "omp_test3_bounce.csv");

    double KE = compute_kinetic_energy(P);
    std::cout << "  Final z  = " << P[0].pos.z << " m\n"
              << "  Final KE = " << KE          << " J\n";
}

// ==============================================================
//  main
// ==============================================================
int main(){
    std::cout << "==============================================\n";
    std::cout << "  DEM Solver  -  OpenMP Parallel Version\n";
    std::cout << "  Available threads: " << omp_get_max_threads() << "\n";
    std::cout << "==============================================\n";

    test_freefall();
    test_bounce();

    correctness_check();
    scaling_study();

    std::cout << "\nDone. Key output files:\n"
              << "  scaling_results.csv  -> speedup/efficiency data\n"
              << "  check_serial.csv / check_par.csv -> correctness comparison\n";
    return 0;
}
