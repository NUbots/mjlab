// Golden-trace generator for the mjlab port of the NUbots quintic walk engine.
//
// Compiles the REAL utility/skill/WalkGenerator.hpp against minimal shims for
// the protobuf messages and NUClear, then runs it through several command
// profiles, dumping the full engine state each control step so the Python port
// can be diffed step by step.
//
// Build and run: see README.md in this directory.

#include <cstdio>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "utility/math/euler.hpp"
#include "utility/skill/WalkGenerator.hpp"

using message::behaviour::state::WalkState;
using utility::input::LimbID;
using utility::math::euler::mat_to_rpy_intrinsic;
using utility::skill::WalkGenerator;

namespace {

    // NUgus tuning from module/skill/Walk/data/config/Walk.yaml.
    WalkGenerator<double>::WalkParameters nugus_parameters(bool only_switch_when_planted) {
        WalkGenerator<double>::WalkParameters p;
        p.step_limits                = Eigen::Vector3d(0.5, 0.2, 0.4);
        p.step_period                = 0.32;
        p.step_height                = 0.085;
        p.step_width                 = 0.27;
        p.step_apex_ratio            = 0.4;
        p.torso_height               = 0.44;
        p.torso_pitch                = 12.0 * M_PI / 180.0;
        p.torso_position_offset      = Eigen::Vector3d(0.01, 0.0, 0.0);
        p.torso_sway_ratio           = 0.5;
        p.torso_sway_offset          = Eigen::Vector3d(0.0, 0.1, 0.0);
        p.torso_start_sway_offset    = Eigen::Vector3d(0.0, 0.1, 0.0);
        p.torso_final_position_ratio = Eigen::Vector3d(0.5, 0.5, 1.0);
        p.only_switch_when_planted   = only_switch_when_planted;
        return p;
    }

    void print_pose(const Eigen::Isometry3d& pose) {
        const Eigen::Vector3d translation = pose.translation();
        const Eigen::Vector3d rpy         = mat_to_rpy_intrinsic(pose.rotation());
        std::printf(",%.15g,%.15g,%.15g,%.15g,%.15g,%.15g",
                    translation.x(),
                    translation.y(),
                    translation.z(),
                    rpy.x(),
                    rpy.y(),
                    rpy.z());
    }

    // Each scenario is a name, a command profile, and whether the engine defers
    // its foot switch to the sensed phase.
    struct Scenario {
        const char* name;
        bool only_switch_when_planted;
    };

    Eigen::Vector3d command_for(const char* name, int step) {
        const double t = step * 0.01;
        if (std::string(name) == "forward") {
            return Eigen::Vector3d(0.2, 0.0, 0.0);
        }
        if (std::string(name) == "start_stop") {
            // Idle, walk, then command zero so the engine stops.
            if (t < 0.25) return Eigen::Vector3d::Zero();
            if (t < 1.75) return Eigen::Vector3d(0.25, 0.0, 0.0);
            return Eigen::Vector3d::Zero();
        }
        if (std::string(name) == "omni") {
            return Eigen::Vector3d(0.15, -0.08, 0.3);
        }
        // "planted": same as forward, but the engine waits on the sensed phase.
        return Eigen::Vector3d(0.2, 0.0, 0.0);
    }

}  // namespace

int main() {
    const Scenario scenarios[] = {
        {"forward", false},
        {"start_stop", false},
        {"omni", false},
        {"planted", true},
    };

    std::printf("scenario,step,state,phase,t");
    std::printf(",torso_x,torso_y,torso_z,torso_roll,torso_pitch,torso_yaw");
    std::printf(",swing_x,swing_y,swing_z,swing_roll,swing_pitch,swing_yaw");
    std::printf(",lfoot_x,lfoot_y,lfoot_z,lfoot_roll,lfoot_pitch,lfoot_yaw");
    std::printf(",rfoot_x,rfoot_y,rfoot_z,rfoot_roll,rfoot_pitch,rfoot_yaw\n");

    for (const Scenario& scenario : scenarios) {
        WalkGenerator<double> generator;
        generator.set_parameters(nugus_parameters(scenario.only_switch_when_planted));
        generator.reset();

        for (int step = 0; step < 250; ++step) {
            // A deterministic sensed phase that alternates every step period,
            // standing in for foot contact. Ignored unless the engine is
            // configured to wait for it.
            const WalkState::Phase sensed =
                ((step / 32) % 2 == 0) ? WalkState::Phase::LEFT : WalkState::Phase::RIGHT;

            const WalkState::State state =
                generator.update(0.01, command_for(scenario.name, step), sensed);

            std::printf("%s,%d,%d,%d,%.15g",
                        scenario.name,
                        step,
                        static_cast<int>(state.value),
                        static_cast<int>(generator.get_phase()),
                        generator.get_time());
            print_pose(generator.get_torso_pose());
            print_pose(generator.get_swing_foot_pose());
            print_pose(generator.get_foot_pose(LimbID::LEFT_LEG));
            print_pose(generator.get_foot_pose(LimbID::RIGHT_LEG));
            std::printf("\n");
        }
    }
    return 0;
}
