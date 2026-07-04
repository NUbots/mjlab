"""Path tracking environments for legged robots.

Instead of a sampled twist command, the policy is given a walk path to
track. The path is what a deployment-side walk path planner produces, so
the trained policy consumes the same interface it will see on the real
robot: a short horizon of upcoming path poses expressed relative to the
robot's own frame.
"""
