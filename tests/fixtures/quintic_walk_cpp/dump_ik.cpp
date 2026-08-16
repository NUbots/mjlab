// Golden-data generator for the mjlab port of the NUbots quintic walk engine.
//
// Compiles the REAL NUbots headers (utility/actuation/InverseKinematics.hpp and
// utility/math/euler.hpp) against minimal shims for the two protobuf messages
// they include, then dumps IK solutions for a deterministic sweep of foot poses
// so the Python port can be diffed against the C++ it was ported from.
//
// Build and run: see README.md in this directory.

#include <cstdint>
#include <cstdio>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "utility/actuation/InverseKinematics.hpp"
#include "utility/math/euler.hpp"

using message::actuation::KinematicsModel;
using utility::actuation::kinematics::calculate_leg_joints;
using utility::input::LimbID;
using utility::math::euler::pos_rpy_to_transform;

namespace {

    // NUgus values from
    // module/actuation/KinematicsConfiguration/data/config/KinematicsConfiguration.yaml.
    KinematicsModel nugus_model() {
        KinematicsModel model;
        model.leg.HIP_OFFSET_X            = 0.00;
        model.leg.HIP_OFFSET_Y            = 0.055;
        model.leg.HIP_OFFSET_Z            = 0.045;
        model.leg.UPPER_LEG_LENGTH        = 0.2;
        model.leg.LOWER_LEG_LENGTH        = 0.2;
        model.leg.FOOT_HEIGHT             = 0.04;
        model.leg.LENGTH_BETWEEN_LEGS     = 2.0 * model.leg.HIP_OFFSET_Y;
        model.leg.LEFT_TO_RIGHT_HIP_YAW   = -1;
        model.leg.LEFT_TO_RIGHT_HIP_ROLL  = -1;
        model.leg.LEFT_TO_RIGHT_HIP_PITCH = 1;
        model.leg.LEFT_TO_RIGHT_KNEE      = 1;
        model.leg.LEFT_TO_RIGHT_ANKLE_PITCH = 1;
        model.leg.LEFT_TO_RIGHT_ANKLE_ROLL  = -1;
        return model;
    }

    // Deterministic LCG (Numerical Recipes constants) so the sweep is
    // reproducible without depending on any standard library RNG.
    struct Lcg {
        std::uint32_t state = 12345u;
        double next(double lo, double hi) {
            state = 1664525u * state + 1013904223u;
            return lo + (hi - lo) * (static_cast<double>(state >> 8) / 16777216.0);
        }
    };

}  // namespace

int main() {
    const KinematicsModel model = nugus_model();
    Lcg rng;

    std::printf("limb,x,y,z,roll,pitch,yaw,");
    std::printf("hip_yaw,hip_roll,hip_pitch,knee_pitch,ankle_pitch,ankle_roll\n");

    for (int limb_index = 0; limb_index < 2; ++limb_index) {
        const bool left     = limb_index == 0;
        const LimbID limb   = left ? LimbID::LEFT_LEG : LimbID::RIGHT_LEG;
        const double y_sign = left ? 1.0 : -1.0;

        for (int sample = 0; sample < 256; ++sample) {
            // The first block stays inside the NUgus walk stance (torso_height
            // 0.44 m, step_width 0.27 m so feet sit at y = +-0.135, torso pitch
            // 12 deg). The second deliberately leaves it, to cover the
            // over-extension clamp and the ankle-above-waist branch that the
            // walk envelope alone never reaches.
            const bool nominal = sample < 192;

            Eigen::Vector3d position;
            Eigen::Vector3d rpy;
            if (nominal) {
                position = Eigen::Vector3d(rng.next(-0.10, 0.10),
                                           y_sign * rng.next(0.08, 0.19),
                                           rng.next(-0.47, -0.34));
                rpy      = Eigen::Vector3d(rng.next(-0.20, 0.20),
                                      rng.next(-0.35, 0.15),
                                      rng.next(-0.30, 0.30));
            }
            else {
                position = Eigen::Vector3d(rng.next(-0.30, 0.30),
                                           y_sign * rng.next(0.02, 0.35),
                                           rng.next(-0.62, 0.12));
                rpy      = Eigen::Vector3d(rng.next(-0.60, 0.60),
                                      rng.next(-0.80, 0.60),
                                      rng.next(-0.80, 0.80));
            }

            const Eigen::Isometry3d Htf = pos_rpy_to_transform(position, rpy);
            const auto joints           = calculate_leg_joints<double>(model, Htf, limb);

            std::printf("%s,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g",
                        left ? "left" : "right",
                        position.x(),
                        position.y(),
                        position.z(),
                        rpy.x(),
                        rpy.y(),
                        rpy.z());
            for (const auto& joint : joints) {
                std::printf(",%.15g", joint.second);
            }
            std::printf("\n");
        }
    }
    return 0;
}
